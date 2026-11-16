//! Épocas de F3: una ronda del protocolo con un solo aprendiz (el nuestro), registrada en
//! `epocas.jsonl` de forma que un tercero pueda REPETIRLA desde la época 0 y obtener los mismos
//! bytes.
//!
//! Reglas (cada una cierra un hallazgo de la segunda auditoría, 2026-11-16):
//! - La cadena entera se valida al arrancar: hash de cada cuerpo, enlace con el anterior, número
//!   correlativo, modelo de partida = vigente anterior, decisión coherente (M-1).
//! - La época 0 parte SIEMPRE del modelo inicial de la semilla (M-2).
//! - Escritura en tres tiempos: candidato con fsync → línea con fsync → renombrar. Un corte en
//!   cualquier punto se recupera solo al arrancar (M-3). Una línea a medias al final se descarta:
//!   nunca llegó a comprometerse (M-4).
//! - Cerrojo exclusivo sobre el directorio (M-5).
//! - El control (bytes y tamaño) y la configuración son los mismos en todas las épocas (M-6).
//! - Una época no adoptada no reescribe el checkpoint (M-7).
use crate::model::{Cfg, Model, MAX_T};
use crate::root::{self, H};
use crate::sha256::{hex, sha256};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const FORMATO: &str = "FRACTAL-CAV-EPOCA-v2";
pub const MAX_PASOS: u64 = 100_000;
pub const MAX_EVAL_LOTES: usize = 4096;
/// Tope de pasos que `repetir` acepta hacer por un registro ajeno (a la forma de ATHENA, ~5 s/paso).
pub const MAX_PASOS_REPETIR: u64 = 2_000_000;

pub fn semilla_epoca(cfg: &Cfg, epoca: u64) -> u64 {
    let mut b = b"FRACTAL-CAV-EPOCA-v1".to_vec();
    b.extend_from_slice(&cfg.seed.to_le_bytes());
    b.extend_from_slice(&epoca.to_le_bytes());
    u64::from_le_bytes(sha256(&b)[..8].try_into().unwrap())
}

/// Parámetros de un trabajo de épocas. El corpus se parte en entrenamiento y control (los
/// últimos `holdout` bytes, que nunca se entrenan).
pub struct Trabajo<'a> {
    pub preset: &'a str,
    pub cfg: Cfg,
    pub corpus: &'a [u8],
    pub holdout: usize,
    pub pasos: u64,
    pub eval_lotes: usize,
    pub hilos: usize,
}

impl<'a> Trabajo<'a> {
    pub fn validar(&self) -> Result<(), String> {
        if self.pasos == 0 || self.pasos > MAX_PASOS {
            return Err(format!("--pasos debe estar entre 1 y {MAX_PASOS}"));
        }
        if self.eval_lotes == 0 || self.eval_lotes > MAX_EVAL_LOTES {
            return Err(format!("--eval-batches debe estar entre 1 y {MAX_EVAL_LOTES}"));
        }
        if self.cfg.v < 256 || !self.cfg.valid() {
            return Err("configuración no apta para datos por bytes".into());
        }
        let minimo = self.cfg.t.checked_add(2).ok_or("contexto demasiado grande")?;
        let resto = self.corpus.len().checked_sub(self.holdout).ok_or("holdout mayor que el corpus")?;
        if self.holdout <= minimo || resto <= minimo {
            return Err("hace falta un control (--holdout) y un entrenamiento mayores que el contexto".into());
        }
        if !crate::det::entorno_ok() {
            return Err("entorno numérico no apto (FMA o subnormales a cero)".into());
        }
        Ok(())
    }
    pub fn entrenamiento(&self) -> &'a [u8] {
        &self.corpus[..self.corpus.len() - self.holdout]
    }
    pub fn control(&self) -> &'a [u8] {
        &self.corpus[self.corpus.len() - self.holdout..]
    }
}

/// Lo que sale de correr una época: todo determinista.
pub struct Resultado {
    pub antes: H,
    pub candidato: H,
    pub trayectoria: Option<H>,
    pub eval_antes: f64,
    pub eval_despues: f64,
    pub adoptada: bool,
    pub motivo: &'static str,
    pub pesos_candidato: Vec<f32>,
}

/// Corre la época `n` sobre el modelo `m` (con el optimizador a cero). No toca ficheros.
pub fn ejecutar(t: &Trabajo, m: &mut Model, n: u64) -> Resultado {
    m.reset_adam();
    let antes = m.roots();
    let pesos_antes = m.p_flat();
    let eval_antes = m.eval(t.control(), t.eval_lotes);
    let seed = semilla_epoca(&t.cfg, n);
    let mut estados = vec![antes.state];
    let mut motivo = "";
    for s in 1..=t.pasos {
        match m.step(t.entrenamiento(), seed, s) {
            Ok(_) => estados.push(m.roots().state),
            Err(e) => {
                motivo = e;
                break;
            }
        }
    }
    let r = if motivo.is_empty() {
        let candidato = m.roots().model;
        let eval_despues = m.eval(t.control(), t.eval_lotes);
        let adoptada = eval_despues.is_finite() && eval_despues < eval_antes;
        Resultado {
            antes: antes.model,
            candidato,
            trayectoria: Some(root::trajectory_root(&estados)),
            eval_antes,
            eval_despues,
            adoptada,
            motivo: if adoptada { "" } else { "no mejora el control" },
            pesos_candidato: m.p_flat(),
        }
    } else {
        Resultado { antes: antes.model, candidato: antes.model, trayectoria: None, eval_antes, eval_despues: f64::INFINITY, adoptada: false, motivo, pesos_candidato: pesos_antes.clone() }
    };
    if !r.adoptada {
        m.set_p_flat(&pesos_antes);
    }
    m.reset_adam();
    r
}

fn bits(x: f64) -> String {
    if x.is_finite() { format!("{:016x}", x.to_bits()) } else { "no_finito".into() }
}

/// El cuerpo determinista de la línea de la época.
pub fn cuerpo(t: &Trabajo, n: u64, previo: &str, r: &Resultado) -> String {
    let vigente = if r.adoptada { r.candidato } else { r.antes };
    let traj = r.trayectoria.map(|h| hex(&h)).unwrap_or_else(|| "0".repeat(64));
    format!(
        "{{\"formato\":\"{FORMATO}\",\"epoca\":{n},\"previo\":\"{previo}\",\"preset\":\"{}\",\"cfg\":\"{}\",\"corpus_sha256\":\"{}\",\"control_sha256\":\"{}\",\"holdout\":{},\"pasos\":{},\"eval_lotes\":{},\"semilla_lotes\":\"{:016x}\",\"model_root_antes\":\"{}\",\"model_root_candidato\":\"{}\",\"trayectoria_root\":\"{traj}\",\"eval_antes_bits\":\"{}\",\"eval_despues_bits\":\"{}\",\"adoptada\":{},\"model_root_vigente\":\"{}\"}}",
        t.preset, hex(&t.cfg.bytes()), hex(&sha256(t.corpus)), hex(&sha256(t.control())), t.holdout, t.pasos, t.eval_lotes,
        semilla_epoca(&t.cfg, n), hex(&r.antes), hex(&r.candidato), bits(r.eval_antes), bits(r.eval_despues), r.adoptada, hex(&vigente)
    )
}

// ------------------------------------------------------------------ lectura del registro
/// Claves del cuerpo, en el ÚNICO orden válido. Un cuerpo con claves de más, de menos, repetidas
/// o en otro orden se rechaza: dos lectores (este, la firma, el ancla, `JSON.parse`) no pueden
/// leer la misma línea de forma distinta. (Tercera auditoría, M-2.)
pub const CLAVES: [&str; 18] = [
    "formato", "epoca", "previo", "preset", "cfg", "corpus_sha256", "control_sha256", "holdout", "pasos", "eval_lotes",
    "semilla_lotes", "model_root_antes", "model_root_candidato", "trayectoria_root", "eval_antes_bits", "eval_despues_bits",
    "adoptada", "model_root_vigente",
];

/// Lee un cuerpo contra la plantilla exacta. Valores: texto `"[0-9A-Za-z_-]*"`, número o booleano.
pub fn campos(c: &str) -> Option<Vec<&str>> {
    let mut r = c.strip_prefix('{')?.strip_suffix('}')?;
    let mut out = Vec::with_capacity(CLAVES.len());
    for (i, k) in CLAVES.iter().enumerate() {
        r = r.strip_prefix('"')?.strip_prefix(*k)?.strip_prefix("\":")?;
        let (v, resto) = if let Some(t) = r.strip_prefix('"') {
            let fin = t.find('"')?;
            let v = &t[..fin];
            if !v.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') {
                return None;
            }
            (v, &t[fin + 1..])
        } else {
            let fin = r.find(',').unwrap_or(r.len());
            let v = &r[..fin];
            // números canónicos: 0 o sin ceros a la izquierda (como los escribe el aprendiz)
            let num = !v.is_empty() && v.len() <= 20 && v.bytes().all(|b| b.is_ascii_digit()) && (v == "0" || !v.starts_with('0'));
            if !(v == "true" || v == "false" || num) {
                return None;
            }
            (v, &r[fin..])
        };
        out.push(v);
        r = if i + 1 < CLAVES.len() { resto.strip_prefix(',')? } else { resto };
    }
    if r.is_empty() { Some(out) } else { None }
}
fn es_hex64(v: &str) -> bool {
    v.len() == 64 && v.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Una época ya registrada.
pub struct Linea {
    pub cuerpo: String,
    pub hash: String,
    valores: Vec<String>,
}
impl Linea {
    pub fn t(&self, k: &str) -> &str {
        CLAVES.iter().position(|x| *x == k).map(|i| self.valores[i].as_str()).unwrap_or("")
    }
    pub fn n(&self, k: &str) -> u64 {
        self.t(k).parse().unwrap_or(u64::MAX)
    }
    pub fn adoptada(&self) -> bool {
        self.t("adoptada") == "true"
    }
}

/// Una línea entera: `{"cuerpo":<cuerpo>,"cuerpo_sha256":"<64 hex>","medido":{<sin llaves>}}`.
pub fn partir(linea: &str) -> Option<Linea> {
    let r = linea.strip_prefix("{\"cuerpo\":")?;
    let fin = r.find('}')? + 1;
    let (c, r) = r.split_at(fin);
    let r = r.strip_prefix(",\"cuerpo_sha256\":\"")?;
    let (h, r) = (r.get(..64)?, r.get(64..)?);
    let medido = r.strip_prefix("\",\"medido\":{")?.strip_suffix("}}")?;
    if !es_hex64(h) || medido.contains('{') || medido.contains('}') {
        return None;
    }
    let valores: Vec<String> = campos(c)?.into_iter().map(String::from).collect();
    for k in ["previo", "corpus_sha256", "control_sha256", "model_root_antes", "model_root_candidato", "trayectoria_root", "model_root_vigente"] {
        if !es_hex64(&valores[CLAVES.iter().position(|x| *x == k).unwrap()]) {
            return None;
        }
    }
    Some(Linea { cuerpo: c.to_string(), hash: h.to_string(), valores })
}

/// Lee y valida TODA la cadena. `genesis`: raíz del modelo inicial del trabajo (la época 0 debe
/// partir de ella). Devuelve las líneas y cuántos bytes del final son una línea a medias.
pub fn leer(texto_registro: &str, genesis: &H, t: &Trabajo) -> Result<(Vec<Linea>, usize), String> {
    let (completo, cola) = match texto_registro.rfind('\n') {
        Some(i) => (&texto_registro[..=i], texto_registro.len() - i - 1),
        None => ("", texto_registro.len()),
    };
    let mut out: Vec<Linea> = Vec::new();
    let mut previo = "0".repeat(64);
    let mut vigente = hex(genesis);
    let control = hex(&sha256(t.control()));
    let corpus = hex(&sha256(t.corpus));
    let cfg = hex(&t.cfg.bytes());
    if completo.contains('\r') {
        return Err("el registro contiene retornos de carro: no es el que escribió el aprendiz".into());
    }
    for (i, l) in completo.lines().enumerate() {
        let err = |m: &str| format!("época {i}: {m}");
        let li = partir(l).ok_or_else(|| err("línea ilegible o fuera de la plantilla"))?;
        if hex(&sha256(li.cuerpo.as_bytes())) != li.hash {
            return Err(err("cuerpo_sha256 no es el hash del cuerpo"));
        }
        if li.t("formato") != FORMATO {
            return Err(err("formato desconocido"));
        }
        if li.n("epoca") != i as u64 {
            return Err(err("número de época no correlativo"));
        }
        if li.t("previo") != previo {
            return Err(err("no encadena con la época anterior"));
        }
        if li.t("model_root_antes") != vigente {
            return Err(err(if i == 0 { "la época 0 no parte del modelo inicial de la semilla" } else { "no parte del modelo vigente de la época anterior" }));
        }
        let esperado = if li.adoptada() { li.t("model_root_candidato") } else { li.t("model_root_antes") };
        if li.t("model_root_vigente") != esperado {
            return Err(err("el modelo vigente no corresponde a la decisión de adoptar"));
        }
        if li.t("cfg") != cfg || li.t("preset") != t.preset {
            return Err(err("otra configuración: es otro trabajo"));
        }
        if li.t("control_sha256") != control || li.n("holdout") != t.holdout as u64 {
            return Err(err("el control cambió: la regla «solo si mejora el control» dejaría de valer"));
        }
        if li.t("corpus_sha256") != corpus {
            return Err(err("otro corpus: el registro dejaría de poder repetirse"));
        }
        previo = li.hash.clone();
        vigente = li.t("model_root_vigente").to_string();
        out.push(li);
    }
    Ok((out, cola))
}

// ------------------------------------------------------------------ correr una época con ficheros
pub struct Rutas {
    pub registro: PathBuf,
    pub actual: PathBuf,
    pub candidato: PathBuf,
    pub cerrojo: PathBuf,
}
pub fn rutas(dir: &Path) -> Rutas {
    Rutas { registro: dir.join("epocas.jsonl"), actual: dir.join("actual.ckpt"), candidato: dir.join("candidato.ckpt"), cerrojo: dir.join("cerrojo") }
}

fn sync_dir(dir: &Path) {
    if let Ok(d) = File::open(dir) {
        let _ = d.sync_all();
    }
}
fn escribir_nuevo(p: &Path, datos: &[u8]) -> Result<(), String> {
    match std::fs::remove_file(p) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("{}: {e}", p.display())),
    }
    // create_new: nunca sigue un enlace ni pisa un fichero que alguien haya puesto ahí
    let mut f = OpenOptions::new().write(true).create_new(true).open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    f.write_all(datos).and_then(|_| f.sync_all()).map_err(|e| format!("{}: {e}", p.display()))
}

fn cargar(p: &Path, cfg: &Cfg, hilos: usize) -> Result<Option<Model>, String> {
    match std::fs::read(p) {
        Ok(b) => Model::load_ckpt(&b, Some(cfg), hilos).map(Some).map_err(|e| format!("{}: {e}", p.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", p.display())),
    }
}

/// Corre la siguiente época del directorio y devuelve la línea registrada.
pub fn correr(t: &Trabajo, dir: &Path) -> Result<String, String> {
    t.validar()?;
    std::fs::create_dir_all(dir).map_err(|e| format!("dir: {e}"))?;
    let r = rutas(dir);
    let cerrojo = OpenOptions::new().create(true).write(true).truncate(false).open(&r.cerrojo).map_err(|e| format!("cerrojo: {e}"))?;
    cerrojo.try_lock().map_err(|_| "otra época está corriendo sobre este directorio".to_string())?;

    let genesis = Model::new(t.cfg.clone(), 1).roots().model; // se suelta enseguida
    let txt = std::fs::read_to_string(&r.registro).or_else(|e| if e.kind() == std::io::ErrorKind::NotFound { Ok(String::new()) } else { Err(e) }).map_err(|e| format!("registro: {e}"))?;
    let (mut lineas, cola) = leer(&txt, &genesis, t)?;
    if cola > 0 {
        let resto = &txt[txt.len() - cola..];
        let completa = format!("{txt}\n");
        match leer(&completa, &genesis, t) {
            // la última línea estaba entera y solo le faltaba el salto (edición manual): se completa
            Ok((l2, 0)) if l2.len() == lineas.len() + 1 => {
                let mut f = OpenOptions::new().append(true).open(&r.registro).map_err(|e| format!("registro: {e}"))?;
                f.write_all(b"\n").and_then(|_| f.sync_all()).map_err(|e| format!("registro: {e}"))?;
                lineas = l2;
            }
            // una línea a medias nunca llegó a comprometerse: se aparta (no se borra) y se quita
            _ => {
                let aparte = dir.join(format!("epocas.cola-{}.txt", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)));
                escribir_nuevo(&aparte, resto.as_bytes())?;
                let f = OpenOptions::new().write(true).open(&r.registro).map_err(|e| format!("registro: {e}"))?;
                f.set_len((txt.len() - cola) as u64).and_then(|_| f.sync_all()).map_err(|e| format!("registro: {e}"))?;
            }
        }
    }
    let vigente = lineas.last().map(|l| l.t("model_root_vigente").to_string()).unwrap_or_else(|| hex(&genesis));

    // modelo vigente: actual.ckpt, o el candidato si el corte cayó justo antes de renombrar,
    // o el inicial si nunca se adoptó ninguno. Un candidato ilegible es un candidato a medias.
    let actual = cargar(&r.actual, &t.cfg, t.hilos)?;
    let es_vigente = |m: &Model| hex(&m.roots().model) == vigente;
    let mut m = match actual {
        Some(m) if es_vigente(&m) => m,
        otro => match cargar(&r.candidato, &t.cfg, t.hilos).ok().flatten() {
            Some(c) if es_vigente(&c) => {
                std::fs::rename(&r.candidato, &r.actual).map_err(|e| format!("recuperar candidato: {e}"))?;
                sync_dir(dir);
                c
            }
            _ if otro.is_none() && vigente == hex(&genesis) => Model::new(t.cfg.clone(), t.hilos),
            _ => return Err("ni actual.ckpt ni candidato.ckpt son el modelo vigente que dice el registro".into()),
        },
    };
    m.threads = t.hilos.max(1);

    let n = lineas.len() as u64;
    if n >= MAX_T as u64 {
        return Err("demasiadas épocas".into());
    }
    let previo = lineas.last().map(|l| l.hash.clone()).unwrap_or_else(|| "0".repeat(64));
    let t0 = std::time::Instant::now();
    let res = ejecutar(t, &mut m, n);
    let seg = t0.elapsed().as_secs_f64();
    let c = cuerpo(t, n, &previo, &res);
    let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let arq = if cfg!(target_arch = "aarch64") { "aarch64" } else { "x86_64" };
    let linea = format!(
        "{{\"cuerpo\":{c},\"cuerpo_sha256\":\"{}\",\"medido\":{{\"unix\":{unix},\"arq\":\"{arq}\",\"hilos\":{},\"segundos\":{seg:.3},\"eval_antes\":{:.6},\"eval_despues\":{},\"motivo\":\"{}\"}}}}\n",
        hex(&sha256(c.as_bytes())), t.hilos, res.eval_antes,
        if res.eval_despues.is_finite() { format!("{:.6}", res.eval_despues) } else { "null".into() }, res.motivo
    );
    if res.adoptada {
        escribir_nuevo(&r.candidato, &m.save_ckpt())?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(&r.registro).map_err(|e| format!("registro: {e}"))?;
    f.write_all(linea.as_bytes()).and_then(|_| f.sync_all()).map_err(|e| format!("registro: {e}"))?;
    if res.adoptada {
        std::fs::rename(&r.candidato, &r.actual).map_err(|e| format!("renombrar candidato: {e}"))?;
    }
    sync_dir(dir);
    Ok(linea)
}

/// Repite TODAS las épocas desde la 0 y comprueba que cada cuerpo sale idéntico byte a byte.
/// Es lo que hace un tercero con el corpus publicado: no se fía de nada de lo que escribimos.
pub fn repetir(t: &Trabajo, registro: &str) -> Result<(usize, String), String> {
    t.validar()?;
    let mut m = Model::new(t.cfg.clone(), t.hilos);
    let genesis = m.roots().model;
    let (lineas, cola) = leer(registro, &genesis, t)?;
    if cola > 0 {
        return Err("el registro termina en una línea a medias".into());
    }
    let total: u64 = lineas.iter().map(|l| l.n("pasos")).fold(0u64, |a, b| a.saturating_add(b));
    if total > MAX_PASOS_REPETIR {
        return Err(format!("el registro pide {total} pasos en total; el tope de una repetición es {MAX_PASOS_REPETIR}"));
    }
    let mut previo = "0".repeat(64);
    for (i, l) in lineas.iter().enumerate() {
        let t_i = Trabajo { pasos: l.n("pasos"), eval_lotes: l.n("eval_lotes") as usize, cfg: t.cfg.clone(), ..*t };
        t_i.validar().map_err(|e| format!("época {i}: {e}"))?;
        let res = ejecutar(&t_i, &mut m, i as u64);
        let c = cuerpo(&t_i, i as u64, &previo, &res);
        if c != l.cuerpo {
            return Err(format!("época {i}: la repetición NO da los mismos bytes"));
        }
        previo = l.hash.clone();
    }
    let vig = lineas.last().map(|l| l.t("model_root_vigente").to_string()).unwrap_or_else(|| hex(&genesis));
    Ok((lineas.len(), vig))
}
