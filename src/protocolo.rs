//! Protocolo de una ronda de la Cadena de Aprendizaje Verificable, para simularlo en local (F2).
//!
//! Una ronda:
//! 1. Todos parten de los mismos pesos `θ`. Cada identidad recibe SU semilla de lotes
//!    (`semilla_lotes`), así que el trabajo de una identidad no sirve para otra.
//! 2. Cada aprendiz da `h` pasos locales y COMPROMETE la raíz de su trayectoria y sus pesos finales.
//! 3. Después del compromiso, una baliza fija qué pasos se revisan (`muestreo`, Fiat-Shamir).
//! 4. El verificador abre cada paso muestreado: camino Merkle de las dos raíces, checkpoint de
//!    partida, re-ejecución exacta. Devuelve además la raíz del gradiente, que solo conoce quien
//!    re-ejecutó.
//! 5. Los deltas que pasan se agregan con un agregador robusto.
use crate::model::{Cfg, Model};
use crate::root::{self, H};
use crate::sha256::sha256;

pub fn hash_de(parts: &[&[u8]]) -> H {
    let mut b = Vec::new();
    for p in parts {
        b.extend_from_slice(&(p.len() as u32).to_le_bytes());
        b.extend_from_slice(p);
    }
    sha256(&b)
}
fn u64_de(h: &H) -> u64 {
    u64::from_le_bytes(h[..8].try_into().unwrap())
}

/// Semilla de lotes de una identidad en una ronda. Sale de la baliza de la ronda ANTERIOR, de modo
/// que nadie puede elegir sus datos ni reutilizar el trabajo de otra identidad.
pub fn semilla_lotes(baliza_previa: &H, id: u32, ronda: u32) -> u64 {
    u64_de(&hash_de(&[b"FRACTAL-CAV-LOTES-v1", baliza_previa, &id.to_le_bytes(), &ronda.to_le_bytes()]))
}

/// Qué pasos se re-ejecutan por sorteo: `s` índices distintos en [0, h-1), función de la baliza
/// POSTERIOR al compromiso, de la identidad, la ronda, `h`, `s` y la raíz comprometida.
/// El paso `k` va del estado k al k+1.
///
/// Requisito que este código NO puede imponer y que le toca a la cadena: UN solo compromiso por
/// (identidad, ronda), fijado antes de que exista la baliza. Si se puede re-comprometer después,
/// el tramposo muele raíces hasta que el sorteo evita su paso falso.
pub fn muestreo(baliza: &H, id: u32, ronda: u32, traj_root: &H, h: u64, s: usize) -> Vec<u64> {
    // El último paso (h-1) se revisa siempre: se sortea entre los demás.
    let sorteable = h.saturating_sub(1);
    let mut out: Vec<u64> = Vec::new();
    if sorteable == 0 {
        return out;
    }
    let mut ctr = 0u32;
    while out.len() < s.min(sorteable as usize) {
        let x = u64_de(&hash_de(&[
            b"FRACTAL-CAV-MUESTRA-v2", baliza, &id.to_le_bytes(), &ronda.to_le_bytes(),
            &h.to_le_bytes(), &(s as u64).to_le_bytes(), traj_root, &ctr.to_le_bytes(),
        ])) % sorteable;
        if !out.contains(&x) {
            out.push(x);
        }
        ctr += 1;
    }
    out
}

/// Pasos que se re-ejecutan SIEMPRE, sin sorteo: el último. Sin esto, basta falsear el paso final
/// para entregar pesos arbitrarios con todo lo demás honesto, y solo se detecta s/h de las veces.
pub fn pasos_obligatorios(h: u64) -> Vec<u64> {
    if h == 0 { vec![] } else { vec![h - 1] }
}

/// Todo lo que se re-ejecuta de un envío: el paso obligatorio y los sorteados, sin repetir.
pub fn pasos_a_verificar(baliza: &H, id: u32, ronda: u32, traj_root: &H, h: u64, s: usize) -> Vec<u64> {
    let mut out = pasos_obligatorios(h);
    for k in muestreo(baliza, id, ronda, traj_root, h, s) {
        if !out.contains(&k) {
            out.push(k);
        }
    }
    out
}

#[derive(Clone)]
pub enum Conducta {
    Honesto,
    /// No calcula los pasos `s` con `(s + desfase) % cada == 0` (deja los pesos como estaban).
    /// `cada = 1`: no calcula nada. Con `desfase` el perezoso evita el último paso, que se revisa siempre.
    Perezoso { cada: u64, desfase: u64 },
    /// Honesto salvo en el paso `en`, donde sustituye los pesos por `objetivo` y sigue honestamente.
    UnPasoFalso { en: u64, objetivo: Vec<f32> },
}

pub struct Envio {
    pub id: u32,
    pub traj_root: H,
    /// `state_root` de los pasos 0..=h (vacío si la corrida no se compromete).
    pub states: Vec<H>,
    pub final_w: Vec<f32>,
    /// Raíces de los momentos del estado final: atan `final_w` a `states[h]`.
    pub final_m: H,
    pub final_v: H,
    /// Checkpoint de cada estado 0..h (lo que el aprendiz debe poder abrir).
    pub ckpts: Vec<Vec<u8>>,
    /// Algún paso dio un resultado no finito: el envío no es válido para nadie.
    pub no_finito: bool,
}

/// Corre los `h` pasos locales de una identidad. Con `comprometer = false` no calcula raíces ni
/// guarda checkpoints (para medir calidad deprisa); el resultado numérico es el mismo.
pub fn corrida_local(m: &mut Model, id: u32, data: &[u8], seed: u64, h: u64, conducta: &Conducta, comprometer: bool) -> Envio {
    let mut states = Vec::new();
    let mut ckpts = Vec::new();
    let mut no_finito = false;
    for s in 1..=h {
        if comprometer {
            states.push(m.roots().state);
            ckpts.push(m.save_ckpt());
        }
        match conducta {
            Conducta::Perezoso { cada, desfase } if (s + desfase) % cada == 0 => m.t += 1,
            Conducta::UnPasoFalso { en, objetivo } if s == *en => {
                m.set_p_flat(objetivo);
                m.t += 1;
            }
            _ => {
                if no_finito || m.step(data, seed, s).is_err() {
                    no_finito = true;
                    m.t = s as u32;
                }
            }
        }
    }
    let (_, fm, fv) = if comprometer { m.roots_parts() } else { ([0; 32], [0; 32], [0; 32]) };
    if comprometer {
        states.push(m.roots().state);
    }
    Envio { id, traj_root: root::trajectory_root(&states), states, final_w: m.p_flat(), final_m: fm, final_v: fv, ckpts, no_finito }
}

/// Lo que el aprendiz entrega cuando le toca abrir el paso `k`.
pub struct Apertura {
    pub k: u64,
    pub ckpt: Vec<u8>,
    pub state_k: H,
    pub path_k: Vec<Option<H>>,
    pub state_k1: H,
    pub path_k1: Vec<Option<H>>,
}
pub fn abrir(e: &Envio, k: u64) -> Apertura {
    let k_ = k as usize;
    Apertura {
        k,
        ckpt: e.ckpts[k_].clone(),
        state_k: e.states[k_],
        path_k: root::trajectory_path(&e.states, k_),
        state_k1: e.states[k_ + 1],
        path_k1: root::trajectory_path(&e.states, k_ + 1),
    }
}

pub struct Informe {
    pub ok: bool,
    pub motivo: &'static str,
    /// Prueba de re-ejecución: raíz del gradiente del paso, ligada a la identidad del verificador.
    /// No impide que dos verificadores coludidos se pasen el gradiente entero; sí impide copiar
    /// una raíz ya publicada.
    pub prueba: H,
}

/// El verificador: comprueba que las dos raíces están en la trayectoria comprometida, que el
/// checkpoint es el de partida Y TRAE LA CONFIGURACIÓN DE LA RONDA, y que re-ejecutar el paso con
/// la semilla DE ESA IDENTIDAD da la raíz de llegada. Nunca entra en pánico con entrada ajena.
pub fn verificar_paso(cfg: &Cfg, traj_root: &H, h: u64, data: &[u8], seed: u64, verificador: u32, ap: &Apertura) -> Informe {
    let mal = |motivo| Informe { ok: false, motivo, prueba: [0; 32] };
    if h == 0 || h > MAX_H_RONDA || ap.k >= h || cfg.v < 256 || data.len() <= cfg.t + 1 {
        return mal("parámetros de la ronda o del paso fuera de rango");
    }
    let n = h as usize + 1;
    if !root::trajectory_verify(&ap.state_k, ap.k as usize, n, &ap.path_k, traj_root)
        || !root::trajectory_verify(&ap.state_k1, ap.k as usize + 1, n, &ap.path_k1, traj_root)
    {
        return mal("camino Merkle inválido");
    }
    let mut m = match Model::load_ckpt(&ap.ckpt, Some(cfg), 1) {
        Ok(m) => m,
        Err(_) => return mal("checkpoint ilegible, no finito o con otra configuración"),
    };
    if m.t as u64 != ap.k || m.roots().state != ap.state_k {
        return mal("el checkpoint no es el estado comprometido");
    }
    if m.step(data, seed, ap.k + 1).is_err() {
        return mal("el paso da un resultado no finito");
    }
    let ok = m.roots().state == ap.state_k1;
    Informe { ok, motivo: if ok { "" } else { "la re-ejecución no reproduce la raíz" }, prueba: m.grad_root_de(verificador) }
}

/// Pasos locales máximos de una ronda. La cota exacta, medida en h·lr, CRECE con h (1,06 con h=20,
/// 2,43 con h=263, 4,23 con h=1000), mientras que un honesto se mueve cada vez menos (0,2 con
/// h=1000). Por encima de ~260 pasos el tramposo que se pega a la cota superaría al envenenador «al
/// tope» medido en F2 (2,43·h·lr). Cuarta auditoría, N-1.
pub const MAX_H_RONDA: u64 = 250;

/// Mayor |m̂_t| / √v̂_t posible en el paso t de AdamW con el optimizador a cero en t=0
/// (β1 = 0,9, β2 = 0,999). Por Cauchy-Schwarz sobre m_t = (1-β1) Σ β1^(t-k) g_k y
/// v_t = (1-β2) Σ β2^(t-k) g_k², con la corrección de sesgo:
///   B(t) = (1-β1)·√(1-β2^t) / ((1-β1^t)·√(1-β2)) · √((1-ρ^t)/(1-ρ)),   ρ = β1²/β2.
/// B(1) = 1 y B(t) → 7,27. (Derivada por la tercera auditoría, 2026-11-16.)
pub fn cota_adam_paso(t: u64) -> f64 {
    let (b1, b2) = (0.9f64, 0.999f64);
    let rho = b1 * b1 / b2;
    let t = t as i32;
    (1.0 - b1) * (1.0 - b2.powi(t)).sqrt() / ((1.0 - b1.powi(t)) * (1.0 - b2).sqrt()) * ((1.0 - rho.powi(t)) / (1.0 - rho)).sqrt()
}

/// Cota EXACTA del desplazamiento de un peso en una ronda honesta de h pasos que parte de `inicio`:
/// un paso mueve |Δ| ≤ lr·(B(t) + wd·|p|), y |p| ≤ |inicio| + lo ya movido. Un honesto nunca la
/// supera (con un margen de redondeo de f32); un tramposo no puede ir más allá.
pub fn cota_desplazamiento(cfg: &Cfg, h: u64, inicio: f32) -> f32 {
    let (lr, wd) = (cfg.lr as f64, cfg.wd as f64);
    let mut adam = 0.0f64;
    for t in 1..=h {
        adam += cota_adam_paso(t);
    }
    cota_desplazamiento_con(lr * adam, lr, wd, h, inicio)
}
fn cota_desplazamiento_con(base: f64, lr: f64, wd: f64, h: u64, inicio: f32) -> f32 {
    let total = base + lr * wd * h as f64 * (inicio.abs() as f64 + base);
    // + el redondeo de f32: medio ulp por paso sobre la magnitud del peso (cuarta auditoría, N-2)
    let ulp = (h as f64 + 1.0) * f32::EPSILON as f64 * (inicio.abs() as f64 + base);
    (total * 1.001 + ulp + 1e-7) as f32
}

/// Comprobaciones que se hacen SIEMPRE, sin muestreo: la trayectoria empieza en los pesos de la
/// ronda (con el optimizador a cero) y termina en los pesos entregados, que han de ser finitos y
/// estar dentro de `cota_desplazamiento` del inicio en cada coordenada.
///
/// Esa cota es la ÚNICA defensa contra un tramposo que inventa un estado intermedio (por ejemplo
/// el penúltimo) y da honestamente el último paso: el sorteo solo lo detecta s/(h-1) de las
/// veces, y sin cota elegiría los pesos finales a su gusto. Con la cota exacta (h=20: 1,06·h·lr)
/// su delta queda por debajo del envenenador «al tope» del experimento C (2,43·h·lr). La primera
/// versión usaba 4·h·lr·(1+|inicio|), que dejaba al tramposo 1,65 veces más fuerte que ese.
/// Ojo: `final_m` y `final_v` los aporta el aprendiz, así que esto NO ata el último estado al
/// anterior; eso lo hace re-ejecutar el paso h-1 (`pasos_obligatorios`).
pub fn verificar_extremos(cfg: &Cfg, inicio: &[f32], h: u64, e: &Envio) -> bool {
    let np = match cfg.param_count() {
        Some(n) => n,
        None => return false,
    };
    if h == 0 || h > MAX_H_RONDA || inicio.len() != np || e.final_w.len() != np {
        return false;
    }
    if e.final_w.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let mut adam = 0.0f64;
    for t in 1..=h {
        adam += cota_adam_paso(t);
    }
    let (lr, wd) = (cfg.lr as f64, cfg.wd as f64);
    let base = lr * adam;
    let fuera = e.final_w.iter().zip(inicio).any(|(f, i)| (f - i).abs() > cota_desplazamiento_con(base, lr, wd, h, *i));
    if fuera {
        return false;
    }
    if e.states.len() != h as usize + 1 || root::trajectory_root(&e.states) != e.traj_root {
        return false;
    }
    let mut m = Model::new(cfg.clone(), 1);
    m.set_p_flat(inicio);
    if m.roots().state != e.states[0] {
        return false;
    }
    m.set_p_flat(&e.final_w);
    let (pm, _, _) = m.roots_parts();
    root::state_root(h as u32, &pm, &e.final_m, &e.final_v) == e.states[h as usize]
}

// ------------------------------------------------------------------ agregación
pub fn delta(final_w: &[f32], inicio: &[f32]) -> Vec<f32> {
    final_w.iter().zip(inicio).map(|(a, b)| a - b).collect()
}
pub fn recortar_tope(d: &mut [f32], tope: f32) {
    for x in d.iter_mut() {
        if *x > tope {
            *x = tope
        } else if *x < -tope {
            *x = -tope
        } else if x.is_nan() {
            *x = 0.0
        }
    }
}
/// Media. NO es robusta: un solo delta enorme o no finito la arruina. Solo como comparación.
pub fn media(ds: &[&[f32]]) -> Vec<f32> {
    let n = ds.len() as f32;
    (0..ds[0].len())
        .map(|j| {
            let mut s = 0.0f32;
            for d in ds {
                s += d[j];
            }
            s / n
        })
        .collect()
}
/// Mediana ponderada por coordenada. Con pesos iguales es la mediana de siempre (media de los dos
/// centrales si el número es par). Orden total de f32, así que es determinista.
pub fn mediana_ponderada(ds: &[&[f32]], w: &[f32]) -> Vec<f32> {
    let total: f32 = w.iter().sum();
    let half = total / 2.0;
    let mut col: Vec<(f32, f32)> = Vec::with_capacity(ds.len());
    (0..ds[0].len())
        .map(|j| {
            col.clear();
            for (d, wi) in ds.iter().zip(w) {
                col.push((d[j], *wi));
            }
            col.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut cum = 0.0f32;
            for i in 0..col.len() {
                cum += col[i].1;
                if cum > half {
                    return col[i].0;
                }
                if cum == half {
                    return if i + 1 < col.len() { 0.5 * (col[i].0 + col[i + 1].0) } else { col[i].0 };
                }
            }
            col[col.len() - 1].0
        })
        .collect()
}
pub fn mediana(ds: &[&[f32]]) -> Vec<f32> {
    mediana_ponderada(ds, &vec![1.0; ds.len()])
}
/// Media recortada: quita los `r` valores más altos y los `r` más bajos de cada coordenada.
pub fn recortada(ds: &[&[f32]], r: usize) -> Vec<f32> {
    assert!(2 * r < ds.len(), "media recortada: hay que dejar al menos un valor");
    let mut col: Vec<f32> = Vec::with_capacity(ds.len());
    (0..ds[0].len())
        .map(|j| {
            col.clear();
            for d in ds {
                col.push(d[j]);
            }
            col.sort_by(|a, b| a.total_cmp(b));
            let keep = &col[r..col.len() - r];
            let mut s = 0.0f32;
            for x in keep {
                s += x;
            }
            s / keep.len() as f32
        })
        .collect()
}
pub fn coseno(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        ab += *x as f64 * *y as f64;
        aa += *x as f64 * *x as f64;
        bb += *y as f64 * *y as f64;
    }
    if aa == 0.0 || bb == 0.0 {
        return 0.0;
    }
    ab / (aa.sqrt() * bb.sqrt())
}
/// Cúmulos por enlace COMPLETO: un delta entra en un cúmulo solo si su coseno con TODOS los
/// miembros supera `tau`. Con enlace simple, un atacante que envía el centroide de los honestos
/// los funde en un solo cúmulo y les quita el voto (demostrado en la auditoría del 2026-11-16).
/// Sirve para copias casi exactas (`tau` ≈ 0,999); no es una defensa contra Sybil.
pub fn cumulos(ds: &[&[f32]], tau: f64) -> Vec<usize> {
    let n = ds.len();
    // Orden canónico (por el hash de los bytes del delta): el resultado no depende de en qué orden
    // llegaron las identidades.
    let clave = |d: &[f32]| {
        let mut b = Vec::with_capacity(4 * d.len());
        for x in d {
            b.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        sha256(&b)
    };
    let claves: Vec<H> = ds.iter().map(|d| clave(d)).collect();
    let mut orden: Vec<usize> = (0..n).collect();
    orden.sort_by(|a, b| claves[*a].cmp(&claves[*b]).then(a.cmp(b)));
    // dos deltas se parecen si son idénticos bit a bit (también el delta nulo) o si su coseno > tau
    let parecidos = |i: usize, j: usize| claves[i] == claves[j] || coseno(ds[i], ds[j]) > tau;
    let mut c: Vec<usize> = vec![usize::MAX; n];
    let mut grupos: Vec<Vec<usize>> = Vec::new();
    for &i in &orden {
        let mut puesto = false;
        for (g, miembros) in grupos.iter_mut().enumerate() {
            if miembros.iter().all(|j| parecidos(i, *j)) {
                miembros.push(i);
                c[i] = g;
                puesto = true;
                break;
            }
        }
        if !puesto {
            c[i] = grupos.len();
            grupos.push(vec![i]);
        }
    }
    c
}
/// Pesos «un cúmulo, un voto»: cada identidad pesa 1 / tamaño de su cúmulo.
pub fn pesos_por_cumulo(c: &[usize]) -> Vec<f32> {
    c.iter().map(|ci| 1.0 / c.iter().filter(|x| *x == ci).count() as f32).collect()
}
pub fn aplicar(inicio: &[f32], agg: &[f32]) -> Vec<f32> {
    inicio.iter().zip(agg).map(|(a, b)| a + b).collect()
}
