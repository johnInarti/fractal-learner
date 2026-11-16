//! Transformer tipo GPT-2 con backprop escrito a mano, en f32 determinista.
//!
//! Es el modelo de la sonda F1 (`research/f1-determinismo`, gradiente comprobado por diferencias
//! finitas) generalizado a dimensiones de ejecución y con paralelismo de «un dueño por salida».
//! Con la configuración de la sonda y `lanes8 = false` reproduce sus hashes bit a bit
//! (prueba `compat_sonda`), lo que ata este motor a lo ya medido en x86 real.
use crate::det::{dot, exp_det, ln_det, sqrt_det, tanh_det};
use crate::par::{split1, split3};
use crate::root::{self, H};
use crate::sha256::sha256;

#[derive(Clone, Debug, PartialEq)]
pub struct Cfg {
    pub v: usize,  // vocabulario
    pub c: usize,  // d_model
    pub nh: usize, // cabezas
    pub l: usize,  // capas
    pub t: usize,  // contexto
    pub b: usize,  // lote
    pub lanes8: bool,
    pub lr: f32,
    pub wd: f32,
    pub seed: u64,
}

pub const SEED_SONDA: u64 = 0x0F1_DE7E_2026_1115;

/// Versión de la aritmética del motor. Entra en `job_id`, en `model_root` y en `state_root`.
/// Se sube con CUALQUIER cambio que altere un bit de un resultado (constantes de `det`, orden de
/// un bucle, optimizador). Dos versiones nunca comparten raíz: una discrepancia entre ellas no se
/// puede confundir con un fraude.
pub const ENGINE_VERSION: u32 = 1;

impl Cfg {
    /// La configuración exacta de la sonda F1 (118 K parámetros).
    pub fn sonda() -> Self {
        Cfg { v: 256, c: 64, nh: 4, l: 2, t: 32, b: 4, lanes8: false, lr: 2e-3f64 as f32, wd: 0.01f64 as f32, seed: SEED_SONDA }
    }
    /// Pequeña y rápida, con el producto escalar de 8 carriles: para pruebas.
    pub fn mini() -> Self {
        Cfg { v: 256, c: 96, nh: 4, l: 2, t: 32, b: 4, lanes8: true, lr: 2e-3f64 as f32, wd: 0.01f64 as f32, seed: SEED_SONDA }
    }
    /// La FORMA de ATHENA en producción: d=512, 4 capas, 4 cabezas, vocabulario de 16 384.
    /// Los datos de la puerta F1 son bytes (ids < 256); el coste y los tensores son los reales.
    pub fn athena() -> Self {
        Cfg { v: 16384, c: 512, nh: 4, l: 4, t: 64, b: 8, lanes8: true, lr: 3e-4f64 as f32, wd: 0.01f64 as f32, seed: SEED_SONDA }
    }
    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "sonda" => Some(Self::sonda()),
            "mini" => Some(Self::mini()),
            "athena" => Some(Self::athena()),
            _ => None,
        }
    }
    pub fn n(&self) -> usize {
        self.b * self.t
    }
    /// Bytes canónicos de la configuración. El número de hilos NO está: no cambia el resultado.
    pub fn bytes(&self) -> Vec<u8> {
        let mut o = Vec::with_capacity(96);
        o.extend_from_slice(b"FRACTAL-CAV-CFG-v1");
        for x in [self.v, self.c, self.nh, self.l, self.t, self.b] {
            o.extend_from_slice(&(x as u64).to_le_bytes());
        }
        o.extend_from_slice(&ENGINE_VERSION.to_le_bytes());
        o.push(self.lanes8 as u8);
        o.extend_from_slice(&self.lr.to_bits().to_le_bytes());
        o.extend_from_slice(&self.wd.to_bits().to_le_bytes());
        o.extend_from_slice(&self.seed.to_le_bytes());
        o
    }
    /// Lo que define la FUNCIÓN del modelo: la forma, la versión del motor y el modo del producto
    /// escalar (los mismos pesos con otro modo dan logits con otros bits). `model_root` depende de
    /// esto y de los pesos, de nada más: ni semilla, ni lote, ni tasa de aprendizaje, ni identidad.
    pub fn arch_bytes(&self) -> Vec<u8> {
        let mut o = Vec::with_capacity(64);
        o.extend_from_slice(b"FRACTAL-CAV-ARCH-v1");
        o.extend_from_slice(&ENGINE_VERSION.to_le_bytes());
        o.push(self.lanes8 as u8);
        for x in [self.v, self.c, self.nh, self.l, self.t] {
            o.extend_from_slice(&(x as u64).to_le_bytes());
        }
        o
    }
    pub fn from_bytes(b: &[u8]) -> Option<(Self, usize)> {
        let tag = b"FRACTAL-CAV-CFG-v1";
        let len = tag.len() + 48 + 4 + 1 + 4 + 4 + 8;
        if b.len() < len || &b[..tag.len()] != tag {
            return None;
        }
        let mut o = tag.len();
        let mut u = [0usize; 6];
        for x in u.iter_mut() {
            *x = usize::try_from(u64::from_le_bytes(b[o..o + 8].try_into().ok()?)).ok()?;
            o += 8;
        }
        if u32::from_le_bytes(b[o..o + 4].try_into().ok()?) != ENGINE_VERSION {
            return None;
        }
        o += 4;
        if b[o] > 1 {
            return None;
        }
        let lanes8 = b[o] != 0;
        o += 1;
        let lr = f32::from_bits(u32::from_le_bytes(b[o..o + 4].try_into().ok()?));
        o += 4;
        let wd = f32::from_bits(u32::from_le_bytes(b[o..o + 4].try_into().ok()?));
        o += 4;
        let seed = u64::from_le_bytes(b[o..o + 8].try_into().ok()?);
        o += 8;
        let cfg = Cfg { v: u[0], c: u[1], nh: u[2], l: u[3], t: u[4], b: u[5], lanes8, lr, wd, seed };
        if !cfg.valid() {
            return None;
        }
        Some((cfg, o))
    }
    /// Número de parámetros con aritmética comprobada (`None` si desborda). Sirve para rechazar
    /// un checkpoint por su tamaño ANTES de reservar memoria.
    pub fn param_count(&self) -> Option<usize> {
        let (v, c, l, t) = (self.v, self.c, self.l, self.t);
        let cc = c.checked_mul(c)?;
        let por_capa = cc.checked_mul(12)?.checked_add(c.checked_mul(13)?)?; // qkv 3c²+3c, ap c²+c, fc 4c²+4c, fp 4c²+c, ln 4c
        v.checked_mul(c)?.checked_add(t.checked_mul(c)?)?.checked_add(l.checked_mul(por_capa)?)?.checked_add(2 * c)
    }
    /// Las activaciones más grandes (logits y atención) caben en 2^31 f32: un checkpoint de pocos
    /// KB no puede pedir terabytes al construir el modelo.
    fn acts_ok(&self) -> bool {
        let lim = 1usize << 31;
        let n = match self.b.checked_mul(self.t) { Some(n) => n, None => return false };
        let logits = n.checked_mul(self.v);
        let att = n.checked_mul(self.nh).and_then(|x| x.checked_mul(self.t));
        let fc = n.checked_mul(self.c).and_then(|x| x.checked_mul(4));
        [logits, att, fc].iter().all(|x| matches!(x, Some(v) if *v <= lim))
    }
    pub fn valid(&self) -> bool {
        self.v >= 2 && self.c >= self.nh && self.nh >= 1 && self.c % self.nh == 0 && self.l >= 1 && self.t >= 2 && self.b >= 1
            && self.v <= 1 << 20 && self.c <= 1 << 14 && self.l <= 256 && self.t <= 1 << 14 && self.b <= 1 << 14
            && self.lr.is_finite() && self.lr >= 0.0 && self.wd.is_finite() && self.wd >= 0.0
            && self.acts_ok()
    }
}

pub const N_TENSORS: usize = 16;

pub struct P {
    pub wte: Vec<f32>,
    pub wpe: Vec<f32>,
    pub ln1w: Vec<f32>,
    pub ln1b: Vec<f32>,
    pub qkvw: Vec<f32>,
    pub qkvb: Vec<f32>,
    pub apw: Vec<f32>,
    pub apb: Vec<f32>,
    pub ln2w: Vec<f32>,
    pub ln2b: Vec<f32>,
    pub fcw: Vec<f32>,
    pub fcb: Vec<f32>,
    pub fpw: Vec<f32>,
    pub fpb: Vec<f32>,
    pub lnfw: Vec<f32>,
    pub lnfb: Vec<f32>,
}

impl P {
    pub fn zeros(k: &Cfg) -> Self {
        let (v, c, l, t) = (k.v, k.c, k.l, k.t);
        let z = |n: usize| vec![0.0f32; n];
        P {
            wte: z(v * c),
            wpe: z(t * c),
            ln1w: z(l * c),
            ln1b: z(l * c),
            qkvw: z(l * 3 * c * c),
            qkvb: z(l * 3 * c),
            apw: z(l * c * c),
            apb: z(l * c),
            ln2w: z(l * c),
            ln2b: z(l * c),
            fcw: z(l * 4 * c * c),
            fcb: z(l * 4 * c),
            fpw: z(l * c * 4 * c),
            fpb: z(l * c),
            lnfw: z(c),
            lnfb: z(c),
        }
    }
    pub fn tensors(&self) -> [&Vec<f32>; N_TENSORS] {
        [
            &self.wte, &self.wpe, &self.ln1w, &self.ln1b, &self.qkvw, &self.qkvb, &self.apw, &self.apb,
            &self.ln2w, &self.ln2b, &self.fcw, &self.fcb, &self.fpw, &self.fpb, &self.lnfw, &self.lnfb,
        ]
    }
    pub fn tensors_mut(&mut self) -> [&mut Vec<f32>; N_TENSORS] {
        [
            &mut self.wte, &mut self.wpe, &mut self.ln1w, &mut self.ln1b, &mut self.qkvw, &mut self.qkvb,
            &mut self.apw, &mut self.apb, &mut self.ln2w, &mut self.ln2b, &mut self.fcw, &mut self.fcb,
            &mut self.fpw, &mut self.fpb, &mut self.lnfw, &mut self.lnfb,
        ]
    }
    pub fn count(&self) -> usize {
        self.tensors().iter().map(|t| t.len()).sum()
    }
}

pub struct Acts {
    pub encoded: Vec<f32>,
    pub ln1: Vec<Vec<f32>>,
    pub ln1_mean: Vec<Vec<f32>>,
    pub ln1_rstd: Vec<Vec<f32>>,
    pub qkv: Vec<Vec<f32>>,
    pub atty: Vec<Vec<f32>>,
    pub preatt: Vec<Vec<f32>>,
    pub att: Vec<Vec<f32>>,
    pub attproj: Vec<Vec<f32>>,
    pub residual2: Vec<Vec<f32>>,
    pub ln2: Vec<Vec<f32>>,
    pub ln2_mean: Vec<Vec<f32>>,
    pub ln2_rstd: Vec<Vec<f32>>,
    pub fch: Vec<Vec<f32>>,
    pub fch_gelu: Vec<Vec<f32>>,
    pub fcproj: Vec<Vec<f32>>,
    pub residual3: Vec<Vec<f32>>,
    pub lnf: Vec<f32>,
    pub lnf_mean: Vec<f32>,
    pub lnf_rstd: Vec<f32>,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub losses: Vec<f32>,
}

impl Acts {
    pub fn new(k: &Cfg) -> Self {
        let (n, c, l) = (k.n(), k.c, k.l);
        let per = |m: usize| (0..l).map(|_| vec![0.0f32; m]).collect::<Vec<_>>();
        let tt = k.b * k.nh * k.t * k.t;
        Acts {
            encoded: vec![0.0; n * c],
            ln1: per(n * c),
            ln1_mean: per(n),
            ln1_rstd: per(n),
            qkv: per(n * 3 * c),
            atty: per(n * c),
            preatt: per(tt),
            att: per(tt),
            attproj: per(n * c),
            residual2: per(n * c),
            ln2: per(n * c),
            ln2_mean: per(n),
            ln2_rstd: per(n),
            fch: per(n * 4 * c),
            fch_gelu: per(n * 4 * c),
            fcproj: per(n * c),
            residual3: per(n * c),
            lnf: vec![0.0; n * c],
            lnf_mean: vec![0.0; n],
            lnf_rstd: vec![0.0; n],
            logits: vec![0.0; n * k.v],
            probs: vec![0.0; n * k.v],
            losses: vec![0.0; n],
        }
    }
    pub fn zero(&mut self) {
        fn f(v: &mut Vec<f32>) {
            for x in v.iter_mut() {
                *x = 0.0;
            }
        }
        fn ff(v: &mut Vec<Vec<f32>>) {
            for w in v.iter_mut() {
                f(w)
            }
        }
        f(&mut self.encoded);
        ff(&mut self.ln1);
        ff(&mut self.ln1_mean);
        ff(&mut self.ln1_rstd);
        ff(&mut self.qkv);
        ff(&mut self.atty);
        ff(&mut self.preatt);
        ff(&mut self.att);
        ff(&mut self.attproj);
        ff(&mut self.residual2);
        ff(&mut self.ln2);
        ff(&mut self.ln2_mean);
        ff(&mut self.ln2_rstd);
        ff(&mut self.fch);
        ff(&mut self.fch_gelu);
        ff(&mut self.fcproj);
        ff(&mut self.residual3);
        f(&mut self.lnf);
        f(&mut self.lnf_mean);
        f(&mut self.lnf_rstd);
        f(&mut self.logits);
        f(&mut self.probs);
        f(&mut self.losses);
    }
}

#[inline]
fn lay(v: &[f32], l: usize, n: usize) -> &[f32] {
    &v[l * n..(l + 1) * n]
}
#[inline]
fn lay_mut(v: &mut [f32], l: usize, n: usize) -> &mut [f32] {
    &mut v[l * n..(l + 1) * n]
}

/// Dimensiones y modo de ejecución que comparten todas las capas.
#[derive(Clone, Copy)]
struct D {
    n: usize,
    c: usize,
    nh: usize,
    t: usize,
    b: usize,
    v: usize,
    l8: bool,
    th: usize,
}

// ------------------------------------------------------------------ capas
fn encoder_fwd(d: D, out: &mut [f32], inp: &[usize], wte: &[f32], wpe: &[f32]) {
    let c = d.c;
    for n in 0..d.n {
        let t = n % d.t;
        let ix = inp[n];
        for i in 0..c {
            out[n * c + i] = wte[ix * c + i] + wpe[t * c + i];
        }
    }
}
fn encoder_bwd(d: D, dwte: &mut [f32], dwpe: &mut [f32], dout: &[f32], inp: &[usize]) {
    let c = d.c;
    for n in 0..d.n {
        let t = n % d.t;
        let ix = inp[n];
        for i in 0..c {
            let g = dout[n * c + i];
            dwte[ix * c + i] += g;
            dwpe[t * c + i] += g;
        }
    }
}

fn ln_fwd(d: D, out: &mut [f32], mean: &mut [f32], rstd: &mut [f32], inp: &[f32], w: &[f32], b: &[f32]) {
    let c = d.c;
    let cf = c as f64 as f32;
    let eps = 1e-5f64 as f32;
    split3(d.th, d.n, out, mean, rstd, |r, o, mn, rs| {
        for (j, n) in r.enumerate() {
            let x = &inp[n * c..(n + 1) * c];
            let mut m = 0.0f32;
            for i in 0..c {
                m += x[i];
            }
            m = m / cf;
            let mut v = 0.0f32;
            for i in 0..c {
                let df = x[i] - m;
                v += df * df;
            }
            v = v / cf;
            let s = 1.0 / sqrt_det(v + eps);
            for i in 0..c {
                o[j * c + i] = s * (x[i] - m) * w[i] + b[i];
            }
            mn[j] = m;
            rs[j] = s;
        }
    });
}
fn ln_bwd(d: D, dinp: &mut [f32], dw: &mut [f32], db: &mut [f32], dout: &[f32], inp: &[f32], w: &[f32], mean: &[f32], rstd: &[f32]) {
    let c = d.c;
    let cf = c as f64 as f32;
    // dw y db acumulan sobre n en orden creciente: un solo hilo (es barato).
    for n in 0..d.n {
        let x = &inp[n * c..(n + 1) * c];
        let g = &dout[n * c..(n + 1) * c];
        let (m, s) = (mean[n], rstd[n]);
        for i in 0..c {
            let norm = (x[i] - m) * s;
            db[i] += g[i];
            dw[i] += norm * g[i];
        }
    }
    split1(d.th, d.n, dinp, |r, di| {
        for (j, n) in r.enumerate() {
            let x = &inp[n * c..(n + 1) * c];
            let g = &dout[n * c..(n + 1) * c];
            let (m, s) = (mean[n], rstd[n]);
            let mut dm = 0.0f32;
            let mut dnm = 0.0f32;
            for i in 0..c {
                let norm = (x[i] - m) * s;
                let dn = w[i] * g[i];
                dm += dn;
                dnm += dn * norm;
            }
            dm = dm / cf;
            dnm = dnm / cf;
            for i in 0..c {
                let norm = (x[i] - m) * s;
                let dn = w[i] * g[i];
                let dval = (dn - dm - norm * dnm) * s;
                di[j * c + i] += dval;
            }
        }
    });
}

fn mm_fwd(d: D, out: &mut [f32], inp: &[f32], w: &[f32], bias: Option<&[f32]>, c: usize, oc: usize) {
    split1(d.th, d.n, out, |r, o| {
        for (j, n) in r.enumerate() {
            let x = &inp[n * c..(n + 1) * c];
            let orow = &mut o[j * oc..(j + 1) * oc];
            for q in 0..oc {
                let init = match bias {
                    Some(b) => b[q],
                    None => 0.0,
                };
                orow[q] = dot(x, &w[q * c..(q + 1) * c], init, d.l8);
            }
        }
    });
}
fn mm_bwd(d: D, dinp: &mut [f32], dw: &mut [f32], db: Option<&mut [f32]>, dout: &[f32], inp: &[f32], w: &[f32], c: usize, oc: usize) {
    let nn = d.n;
    split1(d.th, nn, dinp, |r, blk| {
        for (j, n) in r.enumerate() {
            let g = &dout[n * oc..(n + 1) * oc];
            let di = &mut blk[j * c..(j + 1) * c];
            for q in 0..oc {
                let wr = &w[q * c..(q + 1) * c];
                let gq = g[q];
                for i in 0..c {
                    di[i] += wr[i] * gq;
                }
            }
        }
    });
    split1(d.th, oc, dw, |r, blk| {
        for (j, q) in r.enumerate() {
            let dwr = &mut blk[j * c..(j + 1) * c];
            for n in 0..nn {
                let gq = dout[n * oc + q];
                let x = &inp[n * c..(n + 1) * c];
                for i in 0..c {
                    dwr[i] += x[i] * gq;
                }
            }
        }
    });
    if let Some(db) = db {
        for q in 0..oc {
            let mut s = 0.0f32;
            for n in 0..nn {
                s += dout[n * oc + q];
            }
            db[q] += s;
        }
    }
}

fn attn_fwd(d: D, out: &mut [f32], preatt: &mut [f32], att: &mut [f32], inp: &[f32]) {
    let (c, nh, tt) = (d.c, d.nh, d.t);
    let c3 = 3 * c;
    let hs = c / nh;
    let scale = 1.0 / sqrt_det(hs as f64 as f32);
    split3(d.th, d.b, out, preatt, att, |r, o_blk, pre_blk, att_blk| {
        for (jb, b) in r.enumerate() {
            for t in 0..tt {
                for h in 0..nh {
                    let q = b * tt * c3 + t * c3 + h * hs;
                    let pa = jb * nh * tt * tt + h * tt * tt + t * tt;
                    let mut maxv = 0.0f32;
                    for t2 in 0..=t {
                        let k = b * tt * c3 + t2 * c3 + h * hs + c;
                        let mut v = dot(&inp[q..q + hs], &inp[k..k + hs], 0.0, d.l8);
                        v = v * scale;
                        if t2 == 0 || v > maxv {
                            maxv = v;
                        }
                        pre_blk[pa + t2] = v;
                    }
                    let mut sum = 0.0f32;
                    for t2 in 0..=t {
                        let e = exp_det(pre_blk[pa + t2] - maxv);
                        sum += e;
                        att_blk[pa + t2] = e;
                    }
                    let inv = 1.0 / sum;
                    for t2 in 0..tt {
                        att_blk[pa + t2] = if t2 <= t { att_blk[pa + t2] * inv } else { 0.0 };
                    }
                    let o = jb * tt * c + t * c + h * hs;
                    for i in 0..hs {
                        o_blk[o + i] = 0.0;
                    }
                    for t2 in 0..=t {
                        let vv = b * tt * c3 + t2 * c3 + h * hs + 2 * c;
                        let a = att_blk[pa + t2];
                        for i in 0..hs {
                            o_blk[o + i] += a * inp[vv + i];
                        }
                    }
                }
            }
        }
    });
}
fn attn_bwd(d: D, dinp: &mut [f32], dpreatt: &mut [f32], datt: &mut [f32], dout: &[f32], inp: &[f32], att: &[f32]) {
    let (c, nh, tt) = (d.c, d.nh, d.t);
    let c3 = 3 * c;
    let hs = c / nh;
    let scale = 1.0 / sqrt_det(hs as f64 as f32);
    split3(d.th, d.b, dinp, dpreatt, datt, |r, di, dpre, da| {
        for (jb, b) in r.enumerate() {
            let gi = b * tt * c3; // base global (lectura)
            let li = jb * tt * c3; // base local (escritura)
            for t in 0..tt {
                for h in 0..nh {
                    let q = t * c3 + h * hs;
                    let pag = b * nh * tt * tt + h * tt * tt + t * tt;
                    let pal = jb * nh * tt * tt + h * tt * tt + t * tt;
                    let o = b * tt * c + t * c + h * hs;
                    for t2 in 0..=t {
                        let vv = t2 * c3 + h * hs + 2 * c;
                        let a = att[pag + t2];
                        for i in 0..hs {
                            da[pal + t2] += inp[gi + vv + i] * dout[o + i];
                            di[li + vv + i] += a * dout[o + i];
                        }
                    }
                    for t2 in 0..=t {
                        for t3 in 0..=t {
                            let ind = if t2 == t3 { 1.0f32 } else { 0.0 };
                            let local = att[pag + t2] * (ind - att[pag + t3]);
                            dpre[pal + t3] += local * da[pal + t2];
                        }
                    }
                    for t2 in 0..=t {
                        let k = t2 * c3 + h * hs + c;
                        let g = dpre[pal + t2] * scale;
                        for i in 0..hs {
                            di[li + q + i] += inp[gi + k + i] * g;
                            di[li + k + i] += inp[gi + q + i] * g;
                        }
                    }
                }
            }
        }
    });
}

fn gelu_consts() -> (f32, f32, f32, f32) {
    (
        (2.0 / std::f64::consts::PI).sqrt() as f32,
        0.044715f64 as f32,
        0.5f64 as f32,
        (3.0 * 0.044715f64) as f32,
    )
}
fn gelu_fwd(d: D, out: &mut [f32], inp: &[f32]) {
    let (s, k, half, _) = gelu_consts();
    split1(d.th, out.len(), out, |r, o| {
        for (j, i) in r.enumerate() {
            let x = inp[i];
            let cube = k * x * x * x;
            o[j] = half * x * (1.0 + tanh_det(s * (x + cube)));
        }
    });
}
fn gelu_bwd(d: D, dinp: &mut [f32], inp: &[f32], dout: &[f32]) {
    let (s, k, half, k3) = gelu_consts();
    split1(d.th, dinp.len(), dinp, |r, di| {
        for (j, i) in r.enumerate() {
            let x = inp[i];
            let cube = k * x * x * x;
            let th = tanh_det(s * (x + cube));
            let sech2 = 1.0 - th * th;
            let local = half * (1.0 + th) + x * half * sech2 * s * (1.0 + k3 * x * x);
            di[j] += local * dout[i];
        }
    });
}

fn residual_fwd(out: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..out.len() {
        out[i] = a[i] + b[i];
    }
}
fn residual_bwd(da: &mut [f32], db: &mut [f32], dout: &[f32]) {
    for i in 0..dout.len() {
        da[i] += dout[i];
        db[i] += dout[i];
    }
}

/// softmax + entropía cruzada por logsumexp (pérdida = ln(sum) + max - logit_objetivo)
fn softmax_ce_fwd(d: D, probs: &mut [f32], losses: &mut [f32], logits: &[f32], targets: &[usize]) {
    let v = d.v;
    split3(d.th, d.n, probs, losses, &mut [], |r, pr, ls, _| {
        for (j, n) in r.enumerate() {
            let lg = &logits[n * v..(n + 1) * v];
            let mut maxv = lg[0];
            for i in 1..v {
                if lg[i] > maxv {
                    maxv = lg[i];
                }
            }
            let mut sum = 0.0f32;
            for i in 0..v {
                let e = exp_det(lg[i] - maxv);
                pr[j * v + i] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for i in 0..v {
                pr[j * v + i] = pr[j * v + i] * inv;
            }
            ls[j] = ln_det(sum) + maxv - lg[targets[n]];
        }
    });
}
fn softmax_ce_bwd(d: D, dlogits: &mut [f32], probs: &[f32], targets: &[usize]) {
    let v = d.v;
    let dloss = 1.0 / (d.n as f64 as f32);
    split1(d.th, d.n, dlogits, |r, dl| {
        for (j, n) in r.enumerate() {
            for i in 0..v {
                let ind = if i == targets[n] { 1.0f32 } else { 0.0 };
                dl[j * v + i] += (probs[n * v + i] - ind) * dloss;
            }
        }
    });
}

// ------------------------------------------------------------------ modelo
pub struct Model {
    pub cfg: Cfg,
    pub threads: usize,
    pub p: P,
    pub g: P,
    pub m: P,
    pub v: P,
    pub a: Acts,
    pub ga: Acts,
    pub dlogits: Vec<f32>,
    pub t: u32,
}

pub struct SplitMix(pub u64);
impl SplitMix {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// uniforme en [0,1) exacta (53 bits / 2^53)
    pub fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

const MAGIC: &[u8; 8] = b"FLCK1\0\0\0";
/// Tope del contador de pasos en un checkpoint (la corrección de sesgo de Adam recorre `t`).
pub const MAX_T: u32 = 10_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roots {
    pub model: H,
    pub state: H,
}

impl Model {
    pub fn new(cfg: Cfg, threads: usize) -> Self {
        assert!(cfg.valid(), "configuración inválida");
        let mut p = P::zeros(&cfg);
        let mut rng = SplitMix(cfg.seed);
        // uniforme simétrica con desviación 0.02 (proyecciones residuales: 0.02/sqrt(2L))
        let a = 0.02 * 3f64.sqrt();
        let ar = a / ((2 * cfg.l) as f64).sqrt();
        let init = |v: &mut Vec<f32>, amp: f64, rng: &mut SplitMix| {
            for x in v.iter_mut() {
                *x = ((2.0 * rng.unif() - 1.0) * amp) as f32;
            }
        };
        init(&mut p.wte, a, &mut rng);
        init(&mut p.wpe, a, &mut rng);
        init(&mut p.qkvw, a, &mut rng);
        init(&mut p.apw, ar, &mut rng);
        init(&mut p.fcw, a, &mut rng);
        init(&mut p.fpw, ar, &mut rng);
        for x in p.ln1w.iter_mut().chain(p.ln2w.iter_mut()).chain(p.lnfw.iter_mut()) {
            *x = 1.0;
        }
        Model {
            g: P::zeros(&cfg),
            m: P::zeros(&cfg),
            v: P::zeros(&cfg),
            a: Acts::new(&cfg),
            ga: Acts::new(&cfg),
            dlogits: vec![0.0; cfg.n() * cfg.v],
            t: 0,
            p,
            threads: threads.max(1),
            cfg,
        }
    }

    fn d(&self) -> D {
        let k = &self.cfg;
        D { n: k.n(), c: k.c, nh: k.nh, t: k.t, b: k.b, v: k.v, l8: k.lanes8, th: self.threads }
    }

    /// Pasada hacia delante. Devuelve la pérdida media del lote (suma en f64, orden fijo).
    pub fn forward(&mut self, inp: &[usize], tgt: &[usize]) -> f64 {
        let d = self.d();
        let (c, ll) = (d.c, self.cfg.l);
        assert!(inp.len() == d.n && tgt.len() == d.n);
        let p = &self.p;
        let a = &mut self.a;
        encoder_fwd(d, &mut a.encoded, inp, &p.wte, &p.wpe);
        for l in 0..ll {
            let residual: &[f32] = if l == 0 { &a.encoded } else { &a.residual3[l - 1] };
            ln_fwd(d, &mut a.ln1[l], &mut a.ln1_mean[l], &mut a.ln1_rstd[l], residual, lay(&p.ln1w, l, c), lay(&p.ln1b, l, c));
            mm_fwd(d, &mut a.qkv[l], &a.ln1[l], lay(&p.qkvw, l, 3 * c * c), Some(lay(&p.qkvb, l, 3 * c)), c, 3 * c);
            attn_fwd(d, &mut a.atty[l], &mut a.preatt[l], &mut a.att[l], &a.qkv[l]);
            mm_fwd(d, &mut a.attproj[l], &a.atty[l], lay(&p.apw, l, c * c), Some(lay(&p.apb, l, c)), c, c);
            residual_fwd(&mut a.residual2[l], residual, &a.attproj[l]);
            ln_fwd(d, &mut a.ln2[l], &mut a.ln2_mean[l], &mut a.ln2_rstd[l], &a.residual2[l], lay(&p.ln2w, l, c), lay(&p.ln2b, l, c));
            mm_fwd(d, &mut a.fch[l], &a.ln2[l], lay(&p.fcw, l, 4 * c * c), Some(lay(&p.fcb, l, 4 * c)), c, 4 * c);
            gelu_fwd(d, &mut a.fch_gelu[l], &a.fch[l]);
            mm_fwd(d, &mut a.fcproj[l], &a.fch_gelu[l], lay(&p.fpw, l, 4 * c * c), Some(lay(&p.fpb, l, c)), 4 * c, c);
            let (r2, r3) = (&a.residual2[l], &mut a.residual3[l]);
            residual_fwd(r3, r2, &a.fcproj[l]);
        }
        ln_fwd(d, &mut a.lnf, &mut a.lnf_mean, &mut a.lnf_rstd, &a.residual3[ll - 1], &p.lnfw, &p.lnfb);
        mm_fwd(d, &mut a.logits, &a.lnf, &p.wte, None, c, d.v);
        softmax_ce_fwd(d, &mut a.probs, &mut a.losses, &a.logits, tgt);
        let mut s = 0.0f64;
        for x in a.losses.iter() {
            s += *x as f64;
        }
        s / d.n as f64
    }

    pub fn backward(&mut self, inp: &[usize], tgt: &[usize]) {
        let d = self.d();
        let (c, ll) = (d.c, self.cfg.l);
        for t in self.g.tensors_mut() {
            for x in t.iter_mut() {
                *x = 0.0;
            }
        }
        self.ga.zero();
        for x in self.dlogits.iter_mut() {
            *x = 0.0;
        }
        let p = &self.p;
        let a = &self.a;
        let g = &mut self.g;
        let ga = &mut self.ga;
        softmax_ce_bwd(d, &mut self.dlogits, &a.probs, tgt);
        mm_bwd(d, &mut ga.lnf, &mut g.wte, None, &self.dlogits, &a.lnf, &p.wte, c, d.v);
        ln_bwd(d, &mut ga.residual3[ll - 1], &mut g.lnfw, &mut g.lnfb, &ga.lnf, &a.residual3[ll - 1], &p.lnfw, &a.lnf_mean, &a.lnf_rstd);
        for l in (0..ll).rev() {
            let residual: &[f32] = if l == 0 { &a.encoded } else { &a.residual3[l - 1] };
            let dres3 = std::mem::take(&mut ga.residual3[l]);
            residual_bwd(&mut ga.residual2[l], &mut ga.fcproj[l], &dres3);
            mm_bwd(d, &mut ga.fch_gelu[l], lay_mut(&mut g.fpw, l, 4 * c * c), Some(lay_mut(&mut g.fpb, l, c)), &ga.fcproj[l], &a.fch_gelu[l], lay(&p.fpw, l, 4 * c * c), 4 * c, c);
            gelu_bwd(d, &mut ga.fch[l], &a.fch[l], &ga.fch_gelu[l]);
            mm_bwd(d, &mut ga.ln2[l], lay_mut(&mut g.fcw, l, 4 * c * c), Some(lay_mut(&mut g.fcb, l, 4 * c)), &ga.fch[l], &a.ln2[l], lay(&p.fcw, l, 4 * c * c), c, 4 * c);
            ln_bwd(d, &mut ga.residual2[l], lay_mut(&mut g.ln2w, l, c), lay_mut(&mut g.ln2b, l, c), &ga.ln2[l], &a.residual2[l], lay(&p.ln2w, l, c), &a.ln2_mean[l], &a.ln2_rstd[l]);
            ga.residual3[l] = dres3;
            let dresidual: &mut [f32] = if l == 0 { &mut ga.encoded } else { &mut ga.residual3[l - 1] };
            residual_bwd(dresidual, &mut ga.attproj[l], &ga.residual2[l]);
            mm_bwd(d, &mut ga.atty[l], lay_mut(&mut g.apw, l, c * c), Some(lay_mut(&mut g.apb, l, c)), &ga.attproj[l], &a.atty[l], lay(&p.apw, l, c * c), c, c);
            attn_bwd(d, &mut ga.qkv[l], &mut ga.preatt[l], &mut ga.att[l], &ga.atty[l], &a.qkv[l], &a.att[l]);
            mm_bwd(d, &mut ga.ln1[l], lay_mut(&mut g.qkvw, l, 3 * c * c), Some(lay_mut(&mut g.qkvb, l, 3 * c)), &ga.qkv[l], &a.ln1[l], lay(&p.qkvw, l, 3 * c * c), c, 3 * c);
            let dresidual: &mut [f32] = if l == 0 { &mut ga.encoded } else { &mut ga.residual3[l - 1] };
            ln_bwd(d, dresidual, lay_mut(&mut g.ln1w, l, c), lay_mut(&mut g.ln1b, l, c), &ga.ln1[l], residual, lay(&p.ln1w, l, c), &a.ln1_mean[l], &a.ln1_rstd[l]);
        }
        encoder_bwd(d, &mut g.wte, &mut g.wpe, &ga.encoded, inp);
    }

    /// AdamW. Elemento a elemento: cada parámetro tiene un solo hilo dueño.
    pub fn update(&mut self) {
        self.t += 1;
        let b1 = 0.9f64 as f32;
        let b2 = 0.999f64 as f32;
        let omb1 = 1.0 - b1;
        let omb2 = 1.0 - b2;
        let eps = 1e-8f64 as f32;
        let (lr, wd) = (self.cfg.lr, self.cfg.wd);
        let corr = |beta: f32, t: u32| {
            let mut p = 1.0f32;
            for _ in 0..t {
                p = p * beta;
            }
            1.0 - p
        };
        let bc1 = corr(b1, self.t);
        let bc2 = corr(b2, self.t);
        let th = self.threads;
        let ps = self.p.tensors_mut();
        let gs = self.g.tensors();
        let ms = self.m.tensors_mut();
        let vs = self.v.tensors_mut();
        for (((p, g), m), v) in ps.into_iter().zip(gs).zip(ms).zip(vs) {
            split3(th, p.len(), p, m, v, |r, pp, mm, vv| {
                for (j, i) in r.enumerate() {
                    let gi = g[i];
                    mm[j] = b1 * mm[j] + omb1 * gi;
                    vv[j] = b2 * vv[j] + omb2 * gi * gi;
                    let mhat = mm[j] / bc1;
                    let vhat = vv[j] / bc2;
                    let upd = mhat / (sqrt_det(vhat) + eps) + wd * pp[j];
                    pp[j] = pp[j] - lr * upd;
                }
            });
        }
    }

    /// ¿Pérdida, pesos y momentos son todos finitos?
    pub fn finito(&self) -> bool {
        [&self.p, &self.m, &self.v].iter().all(|set| set.tensors().iter().all(|t| t.iter().all(|x| x.is_finite())))
    }

    /// Un paso completo de entrenamiento sobre el lote `s` del corpus, con la semilla de lotes
    /// `seed` (en el protocolo, una por identidad y ronda).
    ///
    /// Devuelve error si el resultado no es finito. IEEE no fija el signo ni la carga de un NaN:
    /// difieren entre x86 y ARM y entre opciones de compilación, así que un estado con NaN no
    /// tiene raíz comparable. La regla es la misma para todos: paso no finito = paso inválido.
    pub fn step(&mut self, data: &[u8], seed: u64, s: u64) -> Result<f64, &'static str> {
        if !crate::det::entorno_ok() {
            return Err("entorno numérico no apto (FMA o subnormales a cero)");
        }
        let (inp, tgt) = batch(&self.cfg, data, seed, s);
        let loss = self.forward(&inp, &tgt);
        self.backward(&inp, &tgt);
        self.update();
        if !loss.is_finite() || !self.finito() {
            return Err("paso no finito");
        }
        Ok(loss)
    }

    /// Pérdida media sobre ventanas FIJAS del conjunto de control (no depende de ninguna semilla).
    pub fn eval(&mut self, control: &[u8], batches: usize) -> f64 {
        let (b, t) = (self.cfg.b, self.cfg.t);
        assert!(control.len() > t + 1, "control demasiado corto");
        let span = control.len() - t - 1;
        let total = batches * b;
        let mut acc = 0.0f64;
        for k in 0..batches {
            let mut inp = Vec::with_capacity(b * t);
            let mut tgt = Vec::with_capacity(b * t);
            for j in 0..b {
                let w = k * b + j;
                let st = if total > 1 { (span as u128 * w as u128 / (total as u128 - 1).max(1)) as usize } else { 0 };
                let st = st.min(span);
                for i in 0..t {
                    inp.push(control[st + i] as usize);
                    tgt.push(control[st + i + 1] as usize);
                }
            }
            acc += self.forward(&inp, &tgt);
        }
        acc / batches as f64
    }

    /// Pérdida media sobre las ventanas del control que empiezan en `starts` (en lotes de B).
    pub fn eval_windows(&mut self, control: &[u8], starts: &[usize]) -> f64 {
        let (b, t) = (self.cfg.b, self.cfg.t);
        assert!(!starts.is_empty() && starts.len() % b == 0, "ventanas: múltiplo del lote");
        let mut acc = 0.0f64;
        for ch in starts.chunks(b) {
            let mut inp = Vec::with_capacity(b * t);
            let mut tgt = Vec::with_capacity(b * t);
            for &st in ch {
                for i in 0..t {
                    inp.push(control[st + i] as usize);
                    tgt.push(control[st + i + 1] as usize);
                }
            }
            acc += self.forward(&inp, &tgt);
        }
        acc / (starts.len() / b) as f64
    }

    pub fn p_flat(&self) -> Vec<f32> {
        let mut o = Vec::with_capacity(self.p.count());
        for t in self.p.tensors() {
            o.extend_from_slice(t);
        }
        o
    }
    pub fn set_p_flat(&mut self, w: &[f32]) {
        assert_eq!(w.len(), self.p.count());
        let mut o = 0;
        for t in self.p.tensors_mut() {
            let n = t.len();
            t.copy_from_slice(&w[o..o + n]);
            o += n;
        }
    }
    /// Pone a cero los momentos de AdamW y el contador (inicio de una ronda del protocolo).
    pub fn reset_adam(&mut self) {
        for set in [&mut self.m, &mut self.v] {
            for t in set.tensors_mut() {
                for x in t.iter_mut() {
                    *x = 0.0;
                }
            }
        }
        self.t = 0;
    }
    /// Raíz del gradiente del último paso. NO es pública ni se deduce del compromiso: solo la
    /// conoce quien re-ejecutó el paso. Es la prueba de trabajo del verificador.
    pub fn grad_root(&self) -> H {
        root::params_root(self.threads, b"FRACTAL-CAV-GRAD-v1", &self.cfg.arch_bytes(), &self.g.tensors())
    }
    /// La misma raíz, ligada a un verificador: su identidad entra en CADA hoja, así que para
    /// calcularla hace falta el gradiente entero, no las raíces que publicó otro.
    pub fn grad_root_de(&self, verificador: u32) -> H {
        let mut sal = b"verificador/".to_vec();
        sal.extend_from_slice(&verificador.to_le_bytes());
        root::params_root_sal(self.threads, b"FRACTAL-CAV-GRAD-v1", &self.cfg.arch_bytes(), &self.g.tensors(), &sal)
    }

    /// (raíz de pesos, raíz de m, raíz de v): los tres componentes de `state_root`.
    pub fn roots_parts(&self) -> (H, H, H) {
        let cfg = self.cfg.arch_bytes();
        let th = self.threads;
        (
            root::params_root(th, b"FRACTAL-CAV-MODEL-v1", &cfg, &self.p.tensors()),
            root::params_root(th, b"FRACTAL-CAV-ADAM-M-v1", &cfg, &self.m.tensors()),
            root::params_root(th, b"FRACTAL-CAV-ADAM-V-v1", &cfg, &self.v.tensors()),
        )
    }
    pub fn roots(&self) -> Roots {
        let (model, m, v) = self.roots_parts();
        Roots { model, state: root::state_root(self.t, &model, &m, &v) }
    }

    /// SHA-256 de todos los pesos concatenados: el hash que publicó la sonda F1.
    pub fn sha256_pesos_plano(&self) -> H {
        let mut out = Vec::with_capacity(4 * self.p.count());
        for t in self.p.tensors() {
            for x in t.iter() {
                out.extend_from_slice(&x.to_bits().to_le_bytes());
            }
        }
        sha256(&out)
    }

    pub fn save_ckpt(&self) -> Vec<u8> {
        let n = self.p.count();
        let mut out = Vec::with_capacity(128 + 12 * n);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.cfg.bytes());
        out.extend_from_slice(&self.t.to_le_bytes());
        for set in [&self.p, &self.m, &self.v] {
            for t in set.tensors() {
                for x in t.iter() {
                    out.extend_from_slice(&x.to_bits().to_le_bytes());
                }
            }
        }
        out
    }

    /// Carga un checkpoint. `esperada`: la configuración de la ronda. El checkpoint lleva la suya
    /// dentro y la escribe quien lo entrega: si no se compara, el tramposo elige su tasa de
    /// aprendizaje, su lote o su aritmética y «verifica» lo que quiera. Solo se pasa `None` para
    /// inspeccionar un checkpoint propio.
    pub fn load_ckpt(b: &[u8], esperada: Option<&Cfg>, threads: usize) -> Result<Self, String> {
        if b.len() < 8 || &b[..8] != MAGIC {
            return Err("no es un checkpoint FLCK1".into());
        }
        let (cfg, used) = Cfg::from_bytes(&b[8..]).ok_or("configuración ilegible o inválida")?;
        if let Some(e) = esperada {
            if cfg.bytes() != e.bytes() {
                return Err("la configuración del checkpoint no es la de la ronda".into());
            }
        }
        let mut o = 8 + used;
        if b.len() < o + 4 {
            return Err("checkpoint truncado".into());
        }
        let t = u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        o += 4;
        if t > MAX_T {
            return Err("contador de pasos fuera de rango".into());
        }
        // el tamaño se comprueba ANTES de construir el modelo: una cabecera sola no reserva memoria
        let need = cfg.param_count().and_then(|n| n.checked_mul(12)).ok_or("configuración demasiado grande")?;
        if b.len() - o != need {
            return Err(format!("tamaño inesperado: {} bytes de datos, se esperaban {}", b.len() - o, need));
        }
        for w in b[o..].chunks_exact(4) {
            if !f32::from_bits(u32::from_le_bytes(w.try_into().unwrap())).is_finite() {
                return Err("el checkpoint contiene valores no finitos".into());
            }
        }
        let mut m = Model::new(cfg, threads);
        debug_assert_eq!(m.p.count() * 12, need);
        m.t = t;
        for set in [&mut m.p, &mut m.m, &mut m.v] {
            for tensor in set.tensors_mut() {
                for x in tensor.iter_mut() {
                    *x = f32::from_bits(u32::from_le_bytes(b[o..o + 4].try_into().unwrap()));
                    o += 4;
                }
            }
        }
        Ok(m)
    }
}

/// Lote del paso `s` (desde 1): función pura de (semilla, s) y del corpus de entrenamiento.
/// No depende del estado previo: cualquier paso suelto se re-ejecuta desde su checkpoint.
pub fn batch(k: &Cfg, data: &[u8], seed: u64, s: u64) -> (Vec<usize>, Vec<usize>) {
    assert!(data.len() > k.t + 1, "corpus demasiado corto");
    let mut rng = SplitMix(seed ^ s.wrapping_mul(0xd1b5_4a32_d192_ed03));
    let mut inp = Vec::with_capacity(k.n());
    let mut tgt = Vec::with_capacity(k.n());
    let span = (data.len() - k.t - 1) as u64;
    for _ in 0..k.b {
        let st = (rng.next() % span) as usize;
        for i in 0..k.t {
            inp.push(data[st + i] as usize);
            tgt.push(data[st + i + 1] as usize);
        }
    }
    (inp, tgt)
}
