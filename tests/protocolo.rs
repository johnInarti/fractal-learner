//! Pruebas del protocolo de una ronda. `cargo test --release`.
use fractal_learner::model::{Cfg, Model};
use fractal_learner::protocolo::*;
use fractal_learner::root;

const DATA: &[u8] = include_bytes!("../data/dominio-publico.txt");
const HP: u64 = 6;

fn envio(id: u32, seed_id: u32, cond: &Conducta) -> (Vec<f32>, Envio, u64) {
    let c = Cfg::mini();
    let inicio = Model::new(c.clone(), 1).p_flat();
    let b = hash_de(&[b"baliza de prueba"]);
    let seed = semilla_lotes(&b, seed_id, 1);
    let mut m = Model::new(c, 1);
    let mut e = corrida_local(&mut m, seed_id, DATA, seed, HP, cond, true);
    e.id = id;
    (inicio, e, semilla_lotes(&b, id, 1))
}
fn pasos_malos(e: &Envio, seed: u64) -> usize {
    (0..HP).filter(|k| !verificar_paso(&Cfg::mini(), &e.traj_root, HP, DATA, seed, 9, &abrir(e, *k)).ok).count()
}

#[test]
fn honesto_pasa_todo() {
    let (inicio, e, seed) = envio(3, 3, &Conducta::Honesto);
    assert!(verificar_extremos(&Cfg::mini(), &inicio, HP, &e));
    assert_eq!(pasos_malos(&e, seed), 0);
}

/// Quien presenta con su identidad el trabajo de otra falla en TODOS los pasos: la semilla de
/// lotes es por identidad. No hace falta ninguna heurística de parecido.
#[test]
fn clon_falla_en_todos_los_pasos() {
    let (inicio, e, seed) = envio(7, 3, &Conducta::Honesto);
    assert!(verificar_extremos(&Cfg::mini(), &inicio, HP, &e), "los extremos del clon son válidos: no bastan");
    assert_eq!(pasos_malos(&e, seed), HP as usize);
}

#[test]
fn perezoso_y_paso_falso() {
    let (_, e, seed) = envio(1, 1, &Conducta::Perezoso { cada: 3, desfase: 0 });
    assert_eq!(pasos_malos(&e, seed), 2);
    let (inicio, _, _) = envio(2, 2, &Conducta::Honesto);
    let objetivo: Vec<f32> = inicio.iter().map(|x| x * 0.5).collect();
    let (_, e, seed) = envio(2, 2, &Conducta::UnPasoFalso { en: 4, objetivo });
    assert_eq!(pasos_malos(&e, seed), 1, "solo el paso sustituido es falso; los siguientes son honestos");
}

/// El aprendiz no puede abrir un estado que no comprometió, ni cambiar los pesos entregados.
#[test]
fn aperturas_y_extremos_atados_al_compromiso() {
    let (inicio, e, seed) = envio(4, 4, &Conducta::Honesto);
    let mut ap = abrir(&e, 2);
    ap.state_k1[0] ^= 1;
    assert!(!verificar_paso(&Cfg::mini(), &e.traj_root, HP, DATA, seed, 9, &ap).ok);
    let mut ap = abrir(&e, 2);
    ap.k = 3; // el mismo checkpoint presentado como otro paso
    assert!(!verificar_paso(&Cfg::mini(), &e.traj_root, HP, DATA, seed, 9, &ap).ok);
    let mut e2 = e;
    e2.final_w[10] += 1.0;
    assert!(!verificar_extremos(&Cfg::mini(), &inicio, HP, &e2));
}

#[test]
fn camino_merkle() {
    for n in [1usize, 2, 3, 5, 7, 8, 21] {
        let st: Vec<[u8; 32]> = (0..n).map(|i| [i as u8 + 1; 32]).collect();
        let r = root::trajectory_root(&st);
        for i in 0..n {
            let p = root::trajectory_path(&st, i);
            assert!(root::trajectory_verify(&st[i], i, n, &p, &r), "n={n} i={i}");
            assert!(!root::trajectory_verify(&st[i], (i + 1) % n.max(2), n, &p, &r) || n == 1, "n={n} i={i} otro índice");
            assert!(!root::trajectory_verify(&[0xee; 32], i, n, &p, &r));
        }
    }
}

#[test]
fn muestreo_distinto_y_dentro_de_rango() {
    let b = hash_de(&[b"x"]);
    let m = muestreo(&b, 1, 1, &[9; 32], 20, 8);
    assert_eq!(m.len(), 8);
    let mut u = m.clone();
    u.sort();
    u.dedup();
    assert_eq!(u.len(), 8);
    assert!(m.iter().all(|k| *k < 20));
    assert_ne!(m, muestreo(&b, 2, 1, &[9; 32], 20, 8), "depende de la identidad");
    assert_ne!(m, muestreo(&b, 1, 2, &[9; 32], 20, 8), "depende de la ronda");
    assert_eq!(muestreo(&b, 1, 1, &[9; 32], 3, 8).len(), 2, "con h=3 solo se sortean los pasos 0 y 1");
    assert!(pasos_a_verificar(&b, 1, 1, &[9; 32], 20, 2).contains(&19), "el último paso se revisa siempre");
    assert!(!m.contains(&19), "el sorteo no gasta muestras en el paso que ya es obligatorio");
    assert_eq!(pasos_a_verificar(&b, 1, 1, &[9; 32], 20, 2).len(), 3);
}

#[test]
fn agregadores() {
    let a = [1.0f32, 10.0];
    let b = [20.0f32, 2.0];
    let c = [-300.0f32, 5.0];
    let ds: Vec<&[f32]> = vec![&a, &b, &c];
    assert_eq!(mediana(&ds), vec![1.0, 5.0]);
    assert_eq!(recortada(&ds, 1), vec![1.0, 5.0]);
    assert_eq!(media(&ds)[1], 17.0 / 3.0);
    let ds4: Vec<&[f32]> = vec![&a, &b, &c, &c];
    assert_eq!(mediana(&ds4)[0], -149.5, "con pesos iguales y número par: media de los dos centrales");
    // dos copias idénticas forman un cúmulo y cuentan como un voto
    let cl = cumulos(&ds4, 0.9);
    assert_eq!(cl[2], cl[3]);
    assert!(cl[0] != cl[1] && cl[1] != cl[2] && cl[0] != cl[2]);
    let w = pesos_por_cumulo(&cl);
    assert_eq!(mediana_ponderada(&ds4, &w)[0], mediana(&ds)[0]);
}

/// Auditoría del 2026-11-16, hallazgo crítico: el checkpoint trae su propia configuración. Un
/// aprendiz que entrena con OTRA tasa de aprendizaje (ascenso de gradiente, o cero) presentaba una
/// trayectoria que se «verificaba» sola en 20 de 20 pasos. Ahora falla en todos.
#[test]
fn otra_configuracion_no_verifica() {
    for (lr, wd, b) in [(0.02f32, 0.01f32, 1usize), (0.0, 0.01, 1), (2e-3, 0.5, 4), (1e-4, 0.01, 4)] {
        let ronda = Cfg::mini();
        let mut suya = ronda.clone();
        suya.lr = lr;
        suya.wd = wd;
        suya.b = b;
        let bz = hash_de(&[b"baliza de prueba"]);
        let seed = semilla_lotes(&bz, 1, 1);
        let inicio = Model::new(ronda.clone(), 1).p_flat();
        let mut m = Model::new(suya, 1);
        let e = corrida_local(&mut m, 1, DATA, seed, HP, &Conducta::Honesto, true);
        let malos = (0..HP).filter(|k| !verificar_paso(&ronda, &e.traj_root, HP, DATA, seed, 9, &abrir(&e, *k)).ok).count();
        assert_eq!(malos, HP as usize, "lr={lr} wd={wd} b={b}");
    }
}

/// Entradas hostiles no tumban al verificador: devuelven «no».
#[test]
fn el_verificador_no_entra_en_panico() {
    let (inicio, e, seed) = envio(4, 4, &Conducta::Honesto);
    let c = Cfg::mini();
    let mut e2 = Envio { id: e.id, traj_root: e.traj_root, states: e.states.clone(), final_w: e.final_w[..10].to_vec(), final_m: e.final_m, final_v: e.final_v, ckpts: vec![], no_finito: false };
    assert!(!verificar_extremos(&c, &inicio, HP, &e2));
    e2.final_w = e.final_w.clone();
    e2.final_w[3] = f32::INFINITY;
    assert!(!verificar_extremos(&c, &inicio, HP, &e2));
    let mut ap = abrir(&e, 2);
    ap.ckpt.truncate(40);
    assert!(!verificar_paso(&c, &e.traj_root, HP, DATA, seed, 9, &ap).ok);
    let mut ap = abrir(&e, 2);
    ap.k = 999;
    assert!(!verificar_paso(&c, &e.traj_root, HP, DATA, seed, 9, &ap).ok);
    assert!(!verificar_paso(&c, &e.traj_root, 0, DATA, seed, 9, &abrir(&e, 2)).ok);
}

/// El último paso se re-ejecuta siempre: falsearlo ya no permite entregar pesos arbitrarios.
#[test]
fn ultimo_paso_falso_se_detecta_siempre() {
    // un objetivo DENTRO de la cota de desplazamiento: los extremos solos no lo distinguen
    let (inicio, _, _) = envio(2, 2, &Conducta::Honesto);
    let objetivo: Vec<f32> = inicio.iter().enumerate().map(|(i, x)| if i % 2 == 0 { x + 0.01 } else { x - 0.01 }).collect();
    let (_, e, seed) = envio(2, 2, &Conducta::UnPasoFalso { en: HP, objetivo });
    assert!(verificar_extremos(&Cfg::mini(), &inicio, HP, &e), "los extremos solos lo dejan pasar");
    let bz = hash_de(&[b"otra baliza"]);
    for ronda in 0..50u32 {
        let pasos = pasos_a_verificar(&bz, 2, ronda, &e.traj_root, HP, 1);
        assert!(pasos.iter().any(|k| !verificar_paso(&Cfg::mini(), &e.traj_root, HP, DATA, seed, 9, &abrir(&e, *k)).ok));
    }
}

/// La raíz de la trayectoria liga el número de estados.
#[test]
fn la_raiz_liga_la_longitud() {
    let st: Vec<[u8; 32]> = (0..5).map(|i| [i as u8 + 1; 32]).collect();
    let r = root::trajectory_root(&st);
    let p = root::trajectory_path(&st, 4);
    assert!(root::trajectory_verify(&st[4], 4, 5, &p, &r));
    for n in [2usize, 3, 4, 6] {
        for i in 0..n {
            for corte in 0..=p.len() {
                assert!(!root::trajectory_verify(&st[4], i, n, &p[corte..], &r), "n={n} i={i}");
            }
        }
    }
}

/// Enlace completo: el «puente» (el centroide de los honestos) no los funde en un cúmulo.
#[test]
fn el_puente_no_quita_votos() {
    let a = [1.0f32, 0.0, 0.0, 0.55];
    let b = [0.0f32, 1.0, 0.0, 0.55];
    let c = [0.0f32, 0.0, 1.0, 0.55];
    let puente = [1.0f32, 1.0, 1.0, 1.65];
    assert!(coseno(&a, &b) < 0.65 && coseno(&a, &puente) > 0.65);
    let ds: Vec<&[f32]> = vec![&a, &b, &c, &puente];
    let cl = cumulos(&ds, 0.65);
    assert!(cl[0] != cl[1] && cl[1] != cl[2] && cl[0] != cl[2], "los honestos siguen separados: {cl:?}");
    // la prueba del verificador depende de quién verifica
    let mut m = Model::new(Cfg::mini(), 1);
    m.step(DATA, 1, 1).unwrap();
    assert_ne!(m.grad_root_de(1), m.grad_root_de(2));
    assert_ne!(m.grad_root_de(1), m.grad_root());
}

/// Segunda auditoría (2026-11-16), hallazgo A-1: estados 0..h-2 honestos, el penúltimo inventado
/// y el último paso honesto. El paso obligatorio pasa y el sorteo lo ve s/(h-1) de las veces; sin
/// cota, los pesos finales quedaban a 1 ulp de un objetivo arbitrario. Con la cota, los extremos
/// lo rechazan.
#[test]
fn penultimo_inventado_no_da_pesos_arbitrarios() {
    let c = Cfg::mini();
    let (inicio, honesto, seed) = envio(5, 5, &Conducta::Honesto);
    assert!(verificar_extremos(&c, &inicio, HP, &honesto), "el honesto queda dentro de la cota");
    let objetivo: Vec<f32> = inicio.iter().map(|x| -x).collect();
    let (_, e, _) = envio(5, 5, &Conducta::UnPasoFalso { en: HP - 1, objetivo });
    assert_eq!(pasos_malos(&e, seed), 1, "solo el paso inventado es falso");
    assert!(verificar_paso(&c, &e.traj_root, HP, DATA, seed, 9, &abrir(&e, HP - 1)).ok, "el paso obligatorio pasa: es honesto");
    assert!(!verificar_extremos(&c, &inicio, HP, &e), "la cota lo rechaza");
}

/// Hallazgo M-9: copias exactas del delta nulo forman un cúmulo, y el resultado no depende del
/// orden de llegada.
#[test]
fn cumulos_canonicos_y_delta_nulo() {
    let z = [0.0f32; 4];
    let a = [1.0f32, 0.0, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0, 0.0];
    let ds: Vec<&[f32]> = vec![&z, &a, &z, &b, &z];
    let cl = cumulos(&ds, 0.999);
    assert!(cl[0] == cl[2] && cl[2] == cl[4], "tres copias del delta nulo: un cúmulo");
    let w1 = pesos_por_cumulo(&cl);
    let al_reves: Vec<&[f32]> = ds.iter().rev().cloned().collect();
    let mut w2 = pesos_por_cumulo(&cumulos(&al_reves, 0.999));
    w2.reverse();
    assert_eq!(w1, w2, "el peso de cada identidad no depende del orden");
}

/// Tercera auditoría: la cota es la exacta. Un honesto nunca la pasa (varios h, varias semillas) y
/// el tramposo que se pega a ella mueve menos de 2,43·h·lr, el envenenador «al tope» de F2.
#[test]
fn cota_exacta_de_desplazamiento() {
    assert!((cota_adam_paso(1) - 1.0).abs() < 1e-12);
    assert!(cota_adam_paso(100_000) < 7.28 && cota_adam_paso(100_000) > 7.2);
    let c = Cfg::mini();
    assert!(cota_desplazamiento(&c, MAX_H_RONDA, 0.0) / (MAX_H_RONDA as f32 * c.lr) < 2.43, "con el h máximo, por debajo del envenenador de F2");
    // pesos grandes y lr pequeña (la de ATHENA): el redondeo de f32 no rechaza a un honesto
    let mut a = Cfg::mini();
    a.lr = 3e-4;
    let mut m = Model::new(a.clone(), 1);
    for x in m.p.wte.iter_mut() {
        *x = 9.0;
    }
    let inicio_g = m.p_flat();
    let e = corrida_local(&mut m, 0, DATA, 1, 1, &Conducta::Honesto, true);
    assert!(verificar_extremos(&a, &inicio_g, 1, &e), "N-2: un honesto con |w|≈9 y lr=3e-4 queda dentro");
    for h in [1u64, 5, 20, 60] {
        for semilla in 0..3u32 {
            let inicio = Model::new(c.clone(), 1).p_flat();
            let b = hash_de(&[b"cota", &semilla.to_le_bytes()]);
            let mut m = Model::new(c.clone(), 1);
            let e = corrida_local(&mut m, semilla, DATA, semilla_lotes(&b, semilla, 1), h, &Conducta::Honesto, true);
            assert!(verificar_extremos(&c, &inicio, h, &e), "un honesto con h={h} queda dentro");
            let peor = e.final_w.iter().zip(&inicio).map(|(f, i)| ((f - i).abs() / cota_desplazamiento(&c, h, *i)) as f64).fold(0.0, f64::max);
            assert!(peor <= 1.0, "h={h}: {peor}");
        }
    }
    let rel = cota_desplazamiento(&c, 20, 0.0) / (20.0 * c.lr);
    assert!(rel < 1.1, "con h=20 la cota es {rel}·h·lr, por debajo del 2,43 del experimento C");
    let inicio = Model::new(c.clone(), 1).p_flat();
    let (_, mut e, _) = envio(1, 1, &Conducta::Honesto);
    e.final_w = inicio.iter().map(|i| i + 1.5 * cota_desplazamiento(&c, HP, *i)).collect();
    assert!(!verificar_extremos(&c, &inicio, HP, &e));
    assert!(!verificar_extremos(&c, &inicio, MAX_H_RONDA + 1, &e), "h fuera del protocolo");
}
