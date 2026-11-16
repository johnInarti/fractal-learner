//! Simulación de la fase F2 de la Cadena de Aprendizaje Verificable, sobre el motor real.
//!   cargo run --release --bin f2_sim -- [A] [B] [C] [D]      (sin argumentos: todos)
//! Versión 2 (tras la revisión de método del 2026-11-16): tres semillas por escenario, control
//! cortado en frontera de documento, comparación con centralizado a igual cómputo, ataque de
//! minoría más fuerte, horizonte largo, y atribución medida en cuota y con varios sorteos.
use fractal_learner::model::{Cfg, Model};
use fractal_learner::protocolo::*;
use fractal_learner::root::H;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const CORPUS: &[u8] = include_bytes!("../../data/dominio-publico.txt");
/// El control es el último documento ENTERO del corpus (docs/SMART-CONTRACT-GUIDE.md).
const HOLD: usize = 39_244;
const N: usize = 8;
const HP: u64 = 20; // pasos locales por ronda
const EVAL_LOTES: usize = 64;
const SEMILLAS: [u32; 3] = [0, 1, 2];
/// Magnitud por coordenada del ataque «al tope», en unidades de pasos × tasa de aprendizaje.
/// Es 1,25 veces el mayor desplazamiento honesto medido en la primera corrida (1,94).
const TOPE_REL: f32 = 2.43;
/// Pérdida de un modelo que no sabe nada: ln 256.
const SIN_SABER: f64 = 5.545;

fn train_data() -> &'static [u8] {
    &CORPUS[..CORPUS.len() - HOLD]
}
fn control() -> &'static [u8] {
    &CORPUS[CORPUS.len() - HOLD..]
}
fn cfg() -> Cfg {
    Cfg::mini()
}
/// Sustituye a una baliza EXTERNA (drand, bloque de Base...). En la simulación es un hash.
fn baliza(semilla: u32, ronda: u32, que: &[u8]) -> H {
    hash_de(&[b"baliza-simulada", &semilla.to_le_bytes(), &ronda.to_le_bytes(), que])
}
fn perdida(w: &[f32]) -> f64 {
    if w.iter().any(|x| !x.is_finite()) {
        return f64::INFINITY;
    }
    let mut m = Model::new(cfg(), 1);
    m.set_p_flat(w);
    let l = m.eval(control(), EVAL_LOTES);
    if l.is_finite() { l } else { f64::INFINITY }
}
fn stats(xs: &[f64]) -> (f64, f64, f64) {
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (m, xs.iter().cloned().fold(f64::INFINITY, f64::min), xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
}
/// media [mínimo – máximo] sobre las semillas; marca lo que queda peor que no saber nada.
fn celda(xs: &[f64]) -> String {
    if xs.iter().any(|x| !x.is_finite()) {
        return "no finito".into();
    }
    let (m, lo, hi) = stats(xs);
    let s = format!("{m:.3} [{lo:.3} – {hi:.3}]");
    if m >= SIN_SABER { format!("{s} peor que sin entrenar") } else { s }
}

// ====================================================================== A: verificación
fn exp_a(out: &mut String) {
    let c = cfg();
    let inicio = Model::new(c.clone(), 1).p_flat();
    let bprev = baliza(0, 0, b"lotes");
    let seed = |id: u32| semilla_lotes(&bprev, id, 1);
    let correr = |id: u32, cond: &Conducta, comp: bool| {
        let mut m = Model::new(c.clone(), 1);
        m.set_p_flat(&inicio);
        let t = Instant::now();
        let e = corrida_local(&mut m, id, train_data(), seed(id), HP, cond, comp);
        (e, t.elapsed().as_secs_f64())
    };
    let _ = correr(0, &Conducta::Honesto, false); // calentar
    let (_, t_sin) = correr(0, &Conducta::Honesto, false);
    let (e0, t_con) = correr(0, &Conducta::Honesto, true);
    let ajeno = e0.final_w.clone(); // los pesos finales de OTRA identidad
    let d0 = delta(&e0.final_w, &inicio);
    let veneno: Vec<f32> = inicio.iter().zip(&d0).map(|(w, d)| w - 5.0 * d).collect();
    let mut envios: Vec<(&str, Envio)> = vec![("honesto", e0)];
    envios.push(("honesto (otra identidad)", correr(1, &Conducta::Honesto, true).0));
    envios.push(("no calcula nada", correr(2, &Conducta::Perezoso { cada: 1, desfase: 0 }, true).0));
    envios.push(("se salta 1 de cada 4 pasos, nunca el último", correr(3, &Conducta::Perezoso { cada: 4, desfase: 1 }, true).0));
    envios.push(("un paso intermedio falso (el 10) y sigue honesto", correr(4, &Conducta::UnPasoFalso { en: 10, objetivo: veneno.clone() }, true).0));
    envios.push(("último paso falso: entrega pesos arbitrarios", correr(6, &Conducta::UnPasoFalso { en: HP, objetivo: veneno }, true).0));
    envios.push(("último paso falso: entrega los pesos de otro", correr(7, &Conducta::UnPasoFalso { en: HP, objetivo: ajeno }, true).0));
    // copia la trayectoria ENTERA de la identidad 0 y la presenta con la suya
    let (mut clon, _) = correr(0, &Conducta::Honesto, true);
    clon.id = 5;
    envios.push(("copia la trayectoria entera de otra identidad", clon));

    out.push_str("## A. Verificación por re-ejecución\n\n");
    out.push_str(&format!(
        "Montaje: {} parámetros, {HP} pasos locales, corpus de entrenamiento de {} bytes. Cada paso de cada envío se re-ejecutó de verdad con el verificador del protocolo para saber cuáles son falsos. «Sorteo» es la probabilidad exacta (hipergeométrica) de que `s` pasos sorteados toquen uno falso. «Protocolo» es lo medido con 20 000 balizas usando la regla completa: el último paso se re-ejecuta siempre, más 2 sorteados.\n\n",
        inicio.len(), train_data().len()
    ));
    out.push_str("| Conducta | Extremos | Pasos falsos | Sorteo s=2 | Sorteo s=8 | Protocolo (último + 2) |\n|---|---|---|---|---|---|\n");
    let mut t_ver = 0.0;
    let mut n_ver = 0;
    for (nombre, e) in &envios {
        let extremos = verificar_extremos(&c, &inicio, HP, e);
        let mut malos = vec![false; HP as usize];
        for k in 0..HP {
            let ap = abrir(e, k);
            let t = Instant::now();
            let inf = verificar_paso(&c, &e.traj_root, HP, train_data(), seed(e.id), 100, &ap);
            t_ver += t.elapsed().as_secs_f64();
            n_ver += 1;
            malos[k as usize] = !inf.ok;
        }
        let b = malos.iter().filter(|x| **x).count() as u64;
        let exacta = |s: u64| {
            let mut p = 1.0f64; // C(h-b, s) / C(h, s)
            for i in 0..s {
                p *= (HP - b).saturating_sub(i) as f64 / (HP - i) as f64;
            }
            1.0 - p
        };
        let ens = 20_000u32;
        let mut hits = 0u32;
        for r in 0..ens {
            let bz = baliza(9, r, b"muestra");
            if pasos_a_verificar(&bz, e.id, r, &e.traj_root, HP, 2).iter().any(|k| malos[*k as usize]) {
                hits += 1;
            }
        }
        out.push_str(&format!(
            "| {nombre} | {} | {b} de {HP} | {:.1} % | {:.1} % | {:.1} % |\n",
            if extremos { "válidos" } else { "INVÁLIDOS" },
            100.0 * exacta(2), 100.0 * exacta(8), 100.0 * hits as f64 / ens as f64
        ));
    }
    let por_paso = t_ver / n_ver as f64;
    out.push_str(&format!(
        "\n**Coste en esta corrida** (una medición; varía con la carga de la máquina). Entrenar {HP} pasos: {:.0} ms. Comprometer cada paso (raíces y checkpoints) añade {:.0} %. Re-ejecutar un paso con sus comprobaciones: {:.1} ms, {:.1} % del cómputo de la ronda; el protocolo «último + 2» cuesta {:.0} %. En un modelo tan pequeño manda el coste de los hashes.\n\n",
        t_sin * 1e3, 100.0 * (t_con - t_sin) / t_sin, por_paso * 1e3, 100.0 * por_paso / t_sin, 300.0 * por_paso / t_sin
    ));
}

// ====================================================================== B y C: calidad y veneno
#[derive(Clone, Copy, PartialEq)]
enum Agg {
    Media,
    Mediana,
    Recortada(usize),
    Cumulos(f64),
}
#[derive(Clone, Copy, PartialEq)]
enum Veneno {
    Ninguno,
    /// −10 × delta honesto de un atacante; todos envían lo mismo.
    Bruto,
    /// −delta propio de cada atacante: misma norma que un honesto, todos distintos.
    ImitaPropio,
    /// tope × −signo(media de los deltas honestos de los PROPIOS atacantes); todos lo mismo.
    SignoPropio,
    /// Lo mismo, pero cada atacante con el signo de su propio delta: todos distintos.
    SignoDistinto,
}
#[derive(Clone)]
struct Esc {
    nombre: String,
    n: usize,
    agg: Agg,
    k: usize,
    veneno: Veneno,
    conservar_adam: bool,
    rondas: u32,
    central: u64, // >0: corrida centralizada de tantos pasos
    lote: usize,  // lote de la corrida centralizada
    semilla: u32,
}
struct Res {
    perdida: f64,
    max_delta_rel: f32,
    cos_honestos: f64,
    seg: f64,
}

fn correr(e: &Esc) -> Res {
    let c = cfg();
    let t0 = Instant::now();
    if e.central > 0 {
        let mut cc = c.clone();
        cc.b = e.lote;
        let mut m = Model::new(cc, if e.lote > 4 { 4 } else { 1 });
        let seed = semilla_lotes(&baliza(e.semilla, 0, b"central"), 0, 0);
        for s in 1..=e.central {
            if m.step(train_data(), seed, s).is_err() {
                break;
            }
        }
        return Res { perdida: perdida(&m.p_flat()), max_delta_rel: 0.0, cos_honestos: 0.0, seg: t0.elapsed().as_secs_f64() };
    }
    let n = e.n;
    let mut theta = Model::new(c.clone(), 1).p_flat();
    let mut modelos: Vec<Model> = (0..n).map(|_| Model::new(c.clone(), 1)).collect();
    let tope = TOPE_REL * HP as f32 * c.lr;
    let mut max_rel = 0.0f32;
    let (mut cos_sum, mut cos_n) = (0.0f64, 0u32);
    for r in 1..=e.rondas {
        let bprev = baliza(e.semilla, r - 1, b"lotes");
        let finales: Vec<Vec<f32>> = std::thread::scope(|s| {
            let hs: Vec<_> = modelos
                .iter_mut()
                .enumerate()
                .map(|(i, m)| {
                    let theta = &theta;
                    let conservar = e.conservar_adam;
                    s.spawn(move || {
                        m.set_p_flat(theta);
                        if !conservar {
                            m.reset_adam();
                        }
                        let env = corrida_local(m, i as u32, train_data(), semilla_lotes(&bprev, i as u32, r), HP, &Conducta::Honesto, false);
                        if env.no_finito { vec![f32::NAN; env.final_w.len()] } else { env.final_w }
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        if finales.iter().any(|f| f[0].is_nan()) {
            theta = vec![f32::NAN; theta.len()]; // los honestos ya no dan un paso finito desde aquí
            break;
        }
        let mut ds: Vec<Vec<f32>> = finales.iter().map(|f| delta(f, &theta)).collect();
        for d in ds.iter().skip(e.k) {
            for x in d {
                max_rel = max_rel.max(x.abs() / (HP as f32 * c.lr));
            }
        }
        if r % 5 == 1 && n - e.k >= 2 {
            cos_sum += coseno(&ds[n - 1], &ds[n - 2]);
            cos_n += 1;
        }
        // los k primeros son envenenadores: calcularon su delta honesto y envían otra cosa
        if e.k > 0 && e.veneno != Veneno::Ninguno {
            let len = ds[0].len();
            let propio: Vec<f32> = (0..len).map(|j| ds[..e.k].iter().map(|d| d[j]).sum::<f32>() / e.k as f32).collect();
            let signo = |x: f32| if x > 0.0 { -tope } else if x < 0.0 { tope } else { 0.0 };
            let d0 = ds[0].clone();
            for i in 0..e.k {
                ds[i] = match e.veneno {
                    Veneno::Ninguno => unreachable!(),
                    Veneno::Bruto => d0.iter().map(|x| -10.0 * x).collect(),
                    Veneno::ImitaPropio => ds[i].iter().map(|x| -x).collect(),
                    Veneno::SignoPropio => propio.iter().map(|x| signo(*x)).collect(),
                    Veneno::SignoDistinto => ds[i].iter().map(|x| signo(*x)).collect(),
                };
            }
        }
        let refs: Vec<&[f32]> = ds.iter().map(|d| d.as_slice()).collect();
        let agg = match e.agg {
            Agg::Media => media(&refs),
            Agg::Mediana => mediana(&refs),
            Agg::Recortada(q) => recortada(&refs, q),
            Agg::Cumulos(tau) => mediana_ponderada(&refs, &pesos_por_cumulo(&cumulos(&refs, tau))),
        };
        theta = aplicar(&theta, &agg);
        if theta.iter().any(|x| !x.is_finite()) {
            break;
        }
    }
    Res { perdida: perdida(&theta), max_delta_rel: max_rel, cos_honestos: if cos_n > 0 { cos_sum / cos_n as f64 } else { 0.0 }, seg: t0.elapsed().as_secs_f64() }
}

fn cola(escs: &[Esc], trabajadores: usize) -> Vec<Res> {
    let idx = AtomicUsize::new(0);
    let res: Mutex<Vec<Option<Res>>> = Mutex::new((0..escs.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..trabajadores {
            s.spawn(|| loop {
                let i = idx.fetch_add(1, Ordering::SeqCst);
                if i >= escs.len() {
                    break;
                }
                let r = correr(&escs[i]);
                eprintln!("  [{}/{}] {} s{} -> {:.3} ({:.0} s)", i + 1, escs.len(), escs[i].nombre, escs[i].semilla, r.perdida, r.seg);
                res.lock().unwrap()[i] = Some(r);
            });
        }
    });
    res.into_inner().unwrap().into_iter().map(|x| x.unwrap()).collect()
}

fn esc(nombre: &str, agg: Agg, k: usize, veneno: Veneno, rondas: u32) -> Esc {
    Esc { nombre: nombre.into(), n: N, agg, k, veneno, conservar_adam: false, rondas, central: 0, lote: 4, semilla: 0 }
}
/// Corre cada escenario con todas las semillas y devuelve, por escenario, sus pérdidas.
fn con_semillas(base: &[Esc], semillas: &[u32]) -> (Vec<Vec<f64>>, Vec<Res>) {
    let mut todos = Vec::new();
    for e in base {
        for s in semillas {
            let mut x = e.clone();
            x.semilla = *s;
            todos.push(x);
        }
    }
    let res = cola(&todos, 3);
    let per: Vec<Vec<f64>> = res.chunks(semillas.len()).map(|c| c.iter().map(|r| r.perdida).collect()).collect();
    (per, res)
}

fn exp_b(out: &mut String) {
    let r = 40u32;
    let mut escs = vec![
        esc("8 aprendices, mediana", Agg::Mediana, 0, Veneno::Ninguno, r),
        esc("8 aprendices, media", Agg::Media, 0, Veneno::Ninguno, r),
        esc("8 aprendices, media recortada (quita 2 y 2)", Agg::Recortada(2), 0, Veneno::Ninguno, r),
    ];
    let mut e = esc("8 aprendices, mediana, conservando el optimizador entre rondas", Agg::Mediana, 0, Veneno::Ninguno, r);
    e.conservar_adam = true;
    escs.push(e);
    let mut e = esc("1 aprendiz con el mismo protocolo (reinicia el optimizador cada 20 pasos)", Agg::Mediana, 0, Veneno::Ninguno, r);
    e.n = 1;
    escs.push(e);
    let mut c1 = esc("centralizado, 800 pasos, lote 4 (el cómputo de UN aprendiz)", Agg::Media, 0, Veneno::Ninguno, 0);
    c1.central = HP * r as u64;
    escs.push(c1.clone());
    let mut c3 = c1.clone();
    c3.nombre = "centralizado, 800 pasos, lote 32 (el cómputo de los 8, en paralelo)".into();
    c3.lote = 32;
    escs.push(c3);
    let mut c2 = c1.clone();
    c2.nombre = "centralizado, 6400 pasos, lote 4 (el cómputo de los 8, en serie)".into();
    c2.central = HP * r as u64 * N as u64;
    escs.push(c2);
    let (per, res) = con_semillas(&escs, &SEMILLAS);
    let inicial = perdida(&Model::new(cfg(), 1).p_flat());
    out.push_str("## B. Calidad frente a centralizado (sin tramposos)\n\n");
    out.push_str(&format!(
        "Montaje: {HP} pasos locales por ronda, {r} rondas, lote 4, misma tasa de aprendizaje en todo. {} semillas de lotes por fila: media [mínimo – máximo]. Pérdida en el control ({HOLD} bytes: un documento entero que no está en el entrenamiento); al empezar es {inicial:.3}.\n\n| Corrida | Muestras de entrenamiento | Pérdida en control |\n|---|---|---|\n",
        SEMILLAS.len()
    ));
    for (e, x) in escs.iter().zip(&per) {
        let muestras = if e.central > 0 { e.central * e.lote as u64 } else { HP * e.rondas as u64 * e.n as u64 * 4 };
        out.push_str(&format!("| {} | {muestras} | {} |\n", e.nombre, celda(x)));
    }
    let max_rel = res.iter().map(|x| x.max_delta_rel).fold(0.0f32, f32::max);
    let cosm = res[..SEMILLAS.len()].iter().map(|x| x.cos_honestos).sum::<f64>() / SEMILLAS.len() as f64;
    out.push_str(&format!(
        "\nEl mayor desplazamiento de un peso en una ronda honesta fue {max_rel:.2} × (pasos × tasa de aprendizaje). El coseno entre los deltas de dos aprendices honestos fue {cosm:.2} de media.\n\n"
    ));
}

fn exp_c(out: &mut String) {
    let r = 25u32;
    let tau = 0.999f64;
    let aggs: Vec<(&str, Agg)> = vec![("media", Agg::Media), ("mediana", Agg::Mediana), ("un voto por cúmulo de copias + mediana", Agg::Cumulos(tau))];
    let venenos = [
        ("bruto: −10 × delta, todos iguales", Veneno::Bruto),
        ("al tope contra el signo, todos iguales", Veneno::SignoPropio),
        ("al tope contra el signo, cada uno distinto", Veneno::SignoDistinto),
        ("imita la norma: −delta propio, cada uno distinto", Veneno::ImitaPropio),
    ];
    let mut escs = Vec::new();
    for (an, a) in &aggs {
        escs.push(esc(&format!("sin veneno | {an}"), *a, 0, Veneno::Ninguno, r));
        for (vn, v) in &venenos {
            for k in [1usize, 3, 5] {
                escs.push(esc(&format!("{vn} | {an} | k={k}"), *a, k, *v, r));
            }
        }
    }
    let n_tabla = escs.len();
    // 5 honestos solos, y el horizonte largo con mediana
    let mut e5 = esc("5 honestos solos | mediana", Agg::Mediana, 0, Veneno::Ninguno, r);
    e5.n = 5;
    escs.push(e5);
    for k in [0usize, 1, 3] {
        escs.push(esc(&format!("60 rondas | mediana | k={k}"), Agg::Mediana, k, if k == 0 { Veneno::Ninguno } else { Veneno::SignoPropio }, 60));
    }
    let (per, _) = con_semillas(&escs, &SEMILLAS);
    out.push_str("## C. Cuánto daña un envenenador que NO es detectado\n\n");
    out.push_str(&format!(
        "Es el peor caso: los `k` envenenadores de {N} pasan la verificación en todas las rondas ({r} rondas). Calculan su delta honesto y envían otro. «Al tope» = {TOPE_REL} × pasos × tasa de aprendizaje en cada coordenada, contra el signo de su propio delta honesto (no necesitan ver a los demás). «Cúmulo»: deltas con coseno > {tau} con todos los miembros cuentan como un voto. Cada celda: pérdida final en el control, media [mínimo – máximo] de {} semillas. Sin saber nada la pérdida es {SIN_SABER}.\n\n",
        SEMILLAS.len()
    ));
    let mut i = 0;
    for (an, _) in &aggs {
        out.push_str(&format!("**Agregador: {an}.** Sin veneno: {}.\n\n| Ataque | k=1 | k=3 | k=5 |\n|---|---|---|---|\n", celda(&per[i])));
        i += 1;
        for (vn, _) in &venenos {
            out.push_str(&format!("| {vn} | {} | {} | {} |\n", celda(&per[i]), celda(&per[i + 1]), celda(&per[i + 2])));
            i += 3;
        }
        out.push('\n');
    }
    assert_eq!(i, n_tabla);
    out.push_str(&format!(
        "**Comparaciones que faltaban.** Cinco honestos solos, sin los tres envenenadores, a {r} rondas: {}.\n\n**Horizonte largo (mediana, ataque «al tope, todos iguales»).**\n\n| Rondas | Sin veneno | k=1 | k=3 |\n|---|---|---|---|\n| 60 | {} | {} | {} |\n\n",
        celda(&per[n_tabla]), celda(&per[n_tabla + 1]), celda(&per[n_tabla + 2]), celda(&per[n_tabla + 3])
    ));
}

// ====================================================================== D: atribución
fn exp_d(out: &mut String) {
    let c = cfg();
    let nh = 6usize; // honestos en la ronda que se atribuye
    let ronda = |theta: &Vec<f32>, r: u32| -> Vec<Vec<f32>> {
        let bprev = baliza(0, r - 1, b"lotes");
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..N)
                .map(|i| {
                    let c = c.clone();
                    s.spawn(move || {
                        let mut m = Model::new(c, 1);
                        m.set_p_flat(theta);
                        let f = corrida_local(&mut m, i as u32, train_data(), semilla_lotes(&bprev, i as u32, r), HP, &Conducta::Honesto, false).final_w;
                        delta(&f, theta)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };
    let mut theta = Model::new(c.clone(), 1).p_flat();
    for r in 1..=5 {
        let ds = ronda(&theta, r);
        let refs: Vec<&[f32]> = ds.iter().map(|d| d.as_slice()).collect();
        theta = aplicar(&theta, &mediana(&refs));
    }
    let t0 = Instant::now();
    let ds = ronda(&theta, 6);
    let t_ronda = t0.elapsed().as_secs_f64();
    let span = control().len() - c.t - 1;
    let sorteo = |d: u32| -> Vec<usize> {
        let bz = baliza(0, 1000 + d, b"control");
        (0..128u32).map(|j| (u64::from_le_bytes(hash_de(&[&bz, &j.to_le_bytes()])[..8].try_into().unwrap()) % span as u64) as usize).collect()
    };
    let l = |w: &[f32], starts: &[usize]| {
        let mut m = Model::new(c.clone(), 1);
        m.set_p_flat(w);
        m.eval_windows(control(), starts)
    };
    // «dejar uno fuera» por identidad con mediana, sobre un sorteo de control
    let loo = |set: &[&[f32]], starts: &[usize]| -> Vec<f64> {
        let base = l(&aplicar(&theta, &mediana(set)), starts);
        (0..set.len())
            .map(|i| {
                let sin: Vec<&[f32]> = set.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, d)| *d).collect();
                l(&aplicar(&theta, &mediana(&sin)), starts) - base
            })
            .collect()
    };
    let t1 = Instant::now();
    let base6: Vec<&[f32]> = ds[..nh].iter().map(|d| d.as_slice()).collect();
    let r0 = loo(&base6, &sorteo(0));
    let t_loo = t1.elapsed().as_secs_f64();
    let ix = (0..nh).max_by(|a, b| r0[*a].total_cmp(&r0[*b])).unwrap();
    let sorteos = 40u32;

    out.push_str("## D. Atribución de la mejora y copias\n\n");
    out.push_str(&format!(
        "Montaje: tras 5 rondas honestas, una ronda con {nh} aprendices honestos. Recompensa de cada identidad = cuánto empeora el control si se la quita («dejar uno fuera», con mediana). El actor X es el que más aporta y presenta 1, 2, 3 u 8 copias exactas de su delta (se supone que pasaron la verificación: es el caso de datos propios de F4). Cada cifra es media ± desviación sobre {sorteos} sorteos de 128 ventanas de control. «Cuota» = parte de X en la suma de recompensas positivas.\n\n| Copias de X | Suma de recompensas de X (nats) | Cuota de X | Sorteos en que X cobra más que con 1 copia | Honestos con recompensa media negativa |\n|---|---|---|---|---|\n"
    ));
    let mut una: Vec<f64> = Vec::new();
    for copias in [1usize, 2, 3, 8] {
        let mut set: Vec<&[f32]> = vec![ds[ix].as_slice()];
        set.extend(ds[..nh].iter().enumerate().filter(|(i, _)| *i != ix).map(|(_, d)| d.as_slice()));
        for _ in 1..copias {
            set.push(ds[ix].as_slice());
        }
        let es_x: Vec<bool> = (0..set.len()).map(|i| i == 0 || i >= nh).collect();
        let filas: Vec<Vec<f64>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..sorteos)
                .map(|d| {
                    let set = &set;
                    let loo = &loo;
                    let sorteo = &sorteo;
                    s.spawn(move || loo(set, &sorteo(d)))
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let sumas: Vec<f64> = filas.iter().map(|r| r.iter().zip(&es_x).filter(|(_, x)| **x).map(|(v, _)| *v).sum()).collect();
        let cuotas: Vec<f64> = filas
            .iter()
            .map(|r| {
                let pos: f64 = r.iter().map(|v| v.max(0.0)).sum();
                let px: f64 = r.iter().zip(&es_x).filter(|(_, x)| **x).map(|(v, _)| v.max(0.0)).sum();
                if pos > 0.0 { px / pos } else { 0.0 }
            })
            .collect();
        let ms = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (m, (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt())
        };
        let (sm, sd) = ms(&sumas);
        let (cm, cd) = ms(&cuotas);
        if copias == 1 {
            una = sumas.clone();
        }
        let gana = sumas.iter().zip(&una).filter(|(a, b)| a > b).count();
        let negativos = (1..nh).filter(|i| filas.iter().map(|r| r[*i]).sum::<f64>() < 0.0).count();
        out.push_str(&format!(
            "| {copias} | {sm:+.4} ± {sd:.4} | {:.0} % ± {:.0} | {} | {negativos} de {} |\n",
            100.0 * cm, 100.0 * cd, if copias == 1 { "—".to_string() } else { format!("{gana} de {sorteos}") }, nh - 1
        ));
    }
    out.push_str(&format!(
        "\n**Coste (una medición).** «Dejar uno fuera» con {nh} identidades son {} evaluaciones: {:.0} ms. Una ronda de entrenamiento de {N} aprendices: {:.0} ms de reloj en paralelo. Un Shapley con 24 permutaciones serían {} evaluaciones, unas {} veces más.\n\n",
        nh + 1, t_loo * 1e3, t_ronda * 1e3, 24 * nh, 24 * nh / (nh + 1)
    ));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let quiere = |x: &str| args.is_empty() || args.iter().any(|a| a == x);
    let mut out = String::new();
    for (letra, f) in [("A", exp_a as fn(&mut String)), ("B", exp_b), ("C", exp_c), ("D", exp_d)] {
        if quiere(letra) {
            eprintln!("{letra}...");
            let mut parte = String::new();
            f(&mut parte);
            std::fs::create_dir_all("runs/f2").ok();
            std::fs::write(format!("runs/f2/{letra}.md"), &parte).ok();
            out.push_str(&parte);
        }
    }
    print!("{out}");
}
