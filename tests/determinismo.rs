//! Pruebas del motor. Correr con `cargo test --release` (en debug la primera tarda minutos).
use fractal_learner::model::{Cfg, Model};
use fractal_learner::sha256::hex;
use fractal_learner::{det, reexecute_step, root, train, Job};

const DATA: &[u8] = include_bytes!("../data/dominio-publico.txt");

#[test]
fn sin_fma() {
    assert!(det::sin_fma());
}

/// Copia pública: la prueba `compat_sonda` del repositorio de origen ata este motor a los hashes de la sonda F1,
/// medidos con un corpus interno que no se publica. Aquí se conserva sólo la parte que no depende de los datos:
/// los pesos iniciales de la sonda.
#[test]
fn sonda_pesos_iniciales() {
    let cfg = Cfg::sonda();
    let m = Model::new(cfg, 1);
    assert!(m.p.count() > 100_000 && m.p.count() < 130_000, "la sonda tiene ~118 K parámetros");
    assert_eq!(hex(&m.sha256_pesos_plano()), "08fd16c50d60d04651219433ac6838ea858ab38af04bcd50e9e4a5b128d0f86f");
}

fn job(cfg: Cfg, steps: u64) -> Job<'static> {
    Job { cfg, corpus: DATA, holdout: 1500, steps, eval_batches: 4 }
}

/// El número de hilos no entra en ningún hash (ni con el producto de 8 carriles ni con el secuencial).
#[test]
fn hilos_no_cambian_nada() {
    for cfg in [Cfg::mini(), Cfg::sonda()] {
        let a = train(&job(cfg.clone(), 6), 1, |_, _| {});
        for th in [2, 3, 7] {
            let b = train(&job(cfg.clone(), 6), th, |_, _| {});
            assert_eq!(a.states, b.states, "hilos={th}");
            assert_eq!(a.loss_bits, b.loss_bits, "hilos={th}");
            assert_eq!(a.trajectory_root, b.trajectory_root);
            assert_eq!(a.eval_after_bits, b.eval_after_bits);
        }
    }
}

/// El verificador re-ejecuta un paso suelto desde su checkpoint y obtiene la raíz comprometida;
/// un solo bit cambiado en el checkpoint (pesos o momentos) da otra raíz de partida.
#[test]
fn reejecutar_un_paso() {
    let j = job(Cfg::mini(), 8);
    let mut ck = Vec::new();
    let com = train(&j, 2, |s, m| {
        if s == 5 {
            ck = m.save_ckpt();
        }
    });
    let (s, before, r) = reexecute_step(&ck, &j.cfg, j.train_data(), j.cfg.seed, 3).unwrap();
    assert_eq!(s, 6);
    assert_eq!(before, com.states[5]);
    assert_eq!(r.state, com.states[6]);
    assert_eq!(r.model, com.models[6]);

    // Un checkpoint alterado SIEMPRE cambia la raíz de partida. La raíz de llegada NO basta:
    // un bit bajo de un momento de Adam puede absorberse por redondeo (b1*m) y dar el mismo
    // estado siguiente. Por eso el verificador comprueba las DOS raíces.
    for pos in [ck.len() / 5, ck.len() / 2, ck.len() - 3, 95, ck.len() - 1] {
        for bit in [0x01u8, 0x80] {
            let mut bad = ck.clone();
            bad[pos] ^= bit;
            if let Ok((_, b2, _)) = reexecute_step(&bad, &j.cfg, j.train_data(), j.cfg.seed, 1) {
                assert_ne!(b2, com.states[5], "pos={pos} bit={bit}: el checkpoint alterado dio la raíz de partida");
            }
        }
    }
    // entrenar con el control incluido (otro corpus de entrenamiento) tampoco coincide
    let (_, _, r3) = reexecute_step(&ck, &j.cfg, DATA, j.cfg.seed, 1).unwrap();
    assert_ne!(r3.state, com.states[6]);
}

/// El motor de 8 carriles aprende: la pérdida en el control (nunca entrenado) baja.
#[test]
fn aprende_en_control() {
    let com = train(&job(Cfg::mini(), 150), 4, |_, _| {});
    let (a, b) = (f64::from_bits(com.eval_before_bits.unwrap()), f64::from_bits(com.eval_after_bits.unwrap()));
    assert!(a > 5.0 && b < a - 1.5, "control antes={a} después={b}");
}

/// 8 carriles y secuencial son reducciones DISTINTAS: comparten trabajo, no hashes.
#[test]
fn carriles_cambia_el_job() {
    let mut c1 = Cfg::mini();
    c1.lanes8 = false;
    let a = train(&job(Cfg::mini(), 3), 1, |_, _| {});
    let b = train(&job(c1, 3), 1, |_, _| {});
    assert_ne!(a.job_id, b.job_id);
    assert_ne!(a.models[0], b.models[0], "el modo del producto escalar es parte de la función del modelo");
    let mut c2 = Cfg::mini();
    c2.seed ^= 1;
    c2.lr = 1e-3;
    c2.b = 2;
    let (m1, m2) = (Model::new(Cfg::mini(), 1), {
        let mut m = Model::new(c2, 1);
        m.set_p_flat(&Model::new(Cfg::mini(), 1).p_flat());
        m
    });
    assert_eq!(m1.roots().model, m2.roots().model, "ni semilla, ni lote, ni tasa de aprendizaje entran en model_root");
    assert_ne!(a.states[1], b.states[1], "otra reducción, otros bits");
}

/// Hallazgo de la revisión del 2026-11-16: un NaN no tiene bits comparables entre arquitecturas.
/// Un paso que no es finito es un ERROR, igual para todos; y una cfg con tasa no finita o
/// negativa no es válida.
#[test]
fn paso_no_finito_es_error() {
    let mut c = Cfg::mini();
    c.lr = 1e30;
    assert!(c.valid());
    let mut m = Model::new(c, 1);
    let r: Vec<bool> = (1..=3).map(|s| m.step(DATA, 1, s).is_ok()).collect();
    assert!(r.contains(&false), "con lr=1e30 algún paso de los tres primeros deja de ser finito: {r:?}");
    for (lr, wd) in [(f32::NAN, 0.0f32), (f32::INFINITY, 0.0), (-1e-3, 0.0), (1e-3, -1.0), (1e-3, f32::NAN)] {
        let mut c = Cfg::mini();
        c.lr = lr;
        c.wd = wd;
        assert!(!c.valid(), "lr={lr} wd={wd}");
    }
    assert!(det::entorno_ok());
    // un checkpoint diminuto no puede pedir activaciones de terabytes
    let mut c = Cfg::mini();
    c.c = 4;
    c.nh = 1;
    c.t = 16384;
    c.b = 16384;
    assert!(!c.valid());
}

#[test]
fn merkle_sin_duplicar() {
    let h = |x: u8| [x; 32];
    assert_ne!(root::merkle(vec![h(1), h(2), h(3)]), root::merkle(vec![h(1), h(2), h(3), h(3)]));
    assert_eq!(root::merkle(vec![h(7)]), h(7));
    assert_ne!(root::trajectory_root(&[h(1), h(2)]), root::trajectory_root(&[h(2), h(1)]));
}

#[test]
fn checkpoint_malformado_no_entra() {
    assert!(Model::load_ckpt(b"nada", None, 1).is_err());
    let m = Model::new(Cfg::mini(), 1);
    let mut ck = m.save_ckpt();
    ck.truncate(ck.len() - 1);
    assert!(Model::load_ckpt(&ck, None, 1).is_err());
    // una cabecera sola, con una forma enorme, se rechaza por tamaño sin reservar memoria
    let mut grande = Cfg::mini();
    grande.c = 1 << 14;
    grande.l = 256;
    let mut cab = b"FLCK1\0\0\0".to_vec();
    cab.extend_from_slice(&grande.bytes());
    cab.extend_from_slice(&0u32.to_le_bytes());
    assert!(Model::load_ckpt(&cab, None, 1).is_err());
    // un valor no finito tampoco entra
    let mut ck = Model::new(Cfg::mini(), 1).save_ckpt();
    let n = ck.len();
    ck[n - 4..].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    assert!(Model::load_ckpt(&ck, None, 1).is_err());
}
