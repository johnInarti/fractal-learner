//! fractal-learner: motor de entrenamiento determinista de la Cadena de Aprendizaje
//! Verificable (fase F1 de `docs/CADENA-APRENDIZAJE-VERIFICABLE-PLAN.md`).
//!
//! Reglas que este crate cumple y que no se negocian:
//! 1. nada de libm del sistema (transcendentales propias en `det`);
//! 2. nada de FMA (rustc no contrae `a*b+c`; `det::entorno_ok` es una prueba de humo en cada paso);
//! 3. nada de fast-math;
//! 4. toda reducción tiene un orden fijado por el código (`det::dot`), y el paralelismo solo
//!    reparte salidas con un único dueño (`par`): el número de hilos no entra en ningún hash.
#[cfg(not(all(target_pointer_width = "64", any(target_arch = "aarch64", target_arch = "x86_64"))))]
compile_error!("fractal-learner solo está medido en aarch64 y x86_64 de 64 bits");

pub mod det;
pub mod epoca;
pub mod model;
pub mod par;
pub mod protocolo;
pub mod root;
pub mod sha256;

use model::{Cfg, Model};
use root::H;

/// Un trabajo de entrenamiento: todo lo que determina la trayectoria, y nada más.
pub struct Job<'a> {
    pub cfg: Cfg,
    /// Corpus completo. Los últimos `holdout` bytes son el control y NUNCA se entrenan.
    pub corpus: &'a [u8],
    pub holdout: usize,
    pub steps: u64,
    pub eval_batches: usize,
}

impl<'a> Job<'a> {
    pub fn train_data(&self) -> &'a [u8] {
        &self.corpus[..self.corpus.len() - self.holdout]
    }
    pub fn control(&self) -> &'a [u8] {
        &self.corpus[self.corpus.len() - self.holdout..]
    }
    /// Identificador del trabajo: configuración + corpus + partición + pasos.
    pub fn id(&self) -> H {
        let mut b = Vec::new();
        b.extend_from_slice(b"FRACTAL-CAV-JOB-v1");
        b.extend_from_slice(&self.cfg.bytes());
        b.extend_from_slice(&sha256::sha256(self.corpus));
        b.extend_from_slice(&(self.holdout as u64).to_le_bytes());
        b.extend_from_slice(&self.steps.to_le_bytes());
        b.extend_from_slice(&(self.eval_batches as u64).to_le_bytes());
        sha256::sha256(&b)
    }
}

/// Lo que el entrenador compromete: es determinista y no contiene ni tiempos ni hilos.
pub struct Commitment {
    pub job_id: H,
    /// `state_root` de los pasos 0..=steps.
    pub states: Vec<H>,
    pub models: Vec<H>,
    /// Bits de la pérdida de lote de cada paso 1..=steps (f64).
    pub loss_bits: Vec<u64>,
    pub trajectory_root: H,
    pub eval_before_bits: Option<u64>,
    pub eval_after_bits: Option<u64>,
}

/// Entrena el trabajo entero. `on_step(paso, modelo)` se llama tras el paso 0 (inicial) y tras
/// cada paso, para guardar checkpoints o medir tiempos sin tocar el resultado.
pub fn train<F: FnMut(u64, &Model)>(job: &Job, threads: usize, mut on_step: F) -> Commitment {
    assert!(det::entorno_ok(), "entorno numérico no apto (FMA o subnormales a cero): hashes no comparables");
    assert!(job.holdout == 0 || job.eval_batches > 0, "con control hace falta al menos un lote de evaluación");
    assert!(job.cfg.v >= 256, "los datos son bytes: el vocabulario debe ser >= 256");
    let mut m = Model::new(job.cfg.clone(), threads);
    let data = job.train_data();
    let eval_before = if job.holdout > 0 { Some(m.eval(job.control(), job.eval_batches).to_bits()) } else { None };
    let r0 = m.roots();
    let mut states = vec![r0.state];
    let mut models = vec![r0.model];
    let mut loss_bits = Vec::with_capacity(job.steps as usize);
    on_step(0, &m);
    for s in 1..=job.steps {
        let loss = m.step(data, job.cfg.seed, s).unwrap_or_else(|e| panic!("paso {s}: {e}; el trabajo no es válido"));
        let r = m.roots();
        states.push(r.state);
        models.push(r.model);
        loss_bits.push(loss.to_bits());
        on_step(s, &m);
    }
    let eval_after = if job.holdout > 0 { Some(m.eval(job.control(), job.eval_batches).to_bits()) } else { None };
    Commitment {
        job_id: job.id(),
        trajectory_root: root::trajectory_root(&states),
        states,
        models,
        loss_bits,
        eval_before_bits: eval_before,
        eval_after_bits: eval_after,
    }
}

/// Re-ejecuta UN paso desde un checkpoint y devuelve (paso re-ejecutado, raíz de partida, raíces
/// resultantes). `cfg` es la configuración ACORDADA del trabajo: el checkpoint que traiga otra se
/// rechaza. `seed` es la semilla de lotes acordada (en el protocolo, la de la identidad).
pub fn reexecute_step(ckpt: &[u8], cfg: &Cfg, train_data: &[u8], seed: u64, threads: usize) -> Result<(u64, H, model::Roots), String> {
    if cfg.v < 256 || train_data.len() <= cfg.t + 1 {
        return Err("configuración o corpus no aptos para datos por bytes".into());
    }
    let mut m = Model::load_ckpt(ckpt, Some(cfg), threads)?;
    let before = m.roots().state;
    let s = m.t as u64 + 1;
    m.step(train_data, seed, s).map_err(|e| e.to_string())?;
    Ok((s, before, m.roots()))
}
