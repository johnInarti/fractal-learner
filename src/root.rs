//! Raíces por contenido: `model_root` (solo pesos), `state_root` (pesos + momentos de AdamW +
//! contador de pasos: lo que hace falta para re-ejecutar el paso siguiente) y la raíz de una
//! trayectoria (lo que el entrenador compromete ANTES de saber qué pasos se revisan).
//!
//! Formato v1:
//! - hoja        = SHA-256(0x00 ‖ trozo de hasta 2^18 f32 en little-endian)
//! - nodo        = SHA-256(0x01 ‖ izquierda ‖ derecha); un nodo sin pareja SUBE tal cual
//!   (no se duplica: duplicar permite dos árboles distintos con la misma raíz)
//! - tensor      = raíz Merkle de sus hojas
//! - parámetros  = SHA-256(etiqueta ‖ cfg ‖ por cada tensor: longitud u64 LE ‖ raíz del tensor)
//! - estado      = SHA-256("FRACTAL-CAV-STATE-v1" ‖ t u32 LE ‖ raíz(p) ‖ raíz(m) ‖ raíz(v))
use crate::par::split_hash;
use crate::sha256::sha256;

pub const LEAF_F32: usize = 1 << 18;
pub type H = [u8; 32];

pub fn merkle(mut level: Vec<H>) -> H {
    if level.is_empty() {
        return sha256(&[0x02]);
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for pair in level.chunks(2) {
            if pair.len() == 2 {
                let mut buf = [0u8; 65];
                buf[0] = 0x01;
                buf[1..33].copy_from_slice(&pair[0]);
                buf[33..65].copy_from_slice(&pair[1]);
                next.push(sha256(&buf));
            } else {
                next.push(pair[0]);
            }
        }
        level = next;
    }
    level[0]
}

pub fn tensor_root(threads: usize, x: &[f32]) -> H {
    tensor_root_sal(threads, x, &[])
}

/// Como `tensor_root`, con una sal dentro de CADA hoja. Quien no tiene el tensor entero no puede
/// calcularla a partir de la raíz sin sal ni de raíces intermedias.
pub fn tensor_root_sal(threads: usize, x: &[f32], sal: &[u8]) -> H {
    let n_leaves = (x.len() + LEAF_F32 - 1) / LEAF_F32;
    let mut leaves = vec![[0u8; 32]; n_leaves];
    split_hash(threads, &mut leaves, |r, out| {
        let mut buf = Vec::with_capacity(1 + sal.len() + 4 * LEAF_F32);
        for (j, li) in r.enumerate() {
            let chunk = &x[li * LEAF_F32..((li + 1) * LEAF_F32).min(x.len())];
            buf.clear();
            buf.push(if sal.is_empty() { 0x00 } else { 0x03 });
            buf.extend_from_slice(sal);
            for v in chunk {
                buf.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            out[j] = sha256(&buf);
        }
    });
    merkle(leaves)
}

pub fn params_root(threads: usize, tag: &[u8], cfg: &[u8], tensors: &[&Vec<f32>]) -> H {
    params_root_sal(threads, tag, cfg, tensors, &[])
}
pub fn params_root_sal(threads: usize, tag: &[u8], cfg: &[u8], tensors: &[&Vec<f32>], sal: &[u8]) -> H {
    let mut buf = Vec::with_capacity(tag.len() + cfg.len() + tensors.len() * 40);
    buf.extend_from_slice(tag);
    buf.extend_from_slice(cfg);
    for t in tensors {
        buf.extend_from_slice(&(t.len() as u64).to_le_bytes());
        buf.extend_from_slice(&tensor_root_sal(threads, t, sal));
    }
    sha256(&buf)
}

pub fn state_root(t: u32, p: &H, m: &H, v: &H) -> H {
    let mut buf = Vec::with_capacity(20 + 4 + 96);
    buf.extend_from_slice(b"FRACTAL-CAV-STATE-v1");
    buf.extend_from_slice(&t.to_le_bytes());
    buf.extend_from_slice(p);
    buf.extend_from_slice(m);
    buf.extend_from_slice(v);
    sha256(&buf)
}

fn sellar_n(n: usize, m: &H) -> H {
    let mut buf = Vec::with_capacity(19 + 8 + 32);
    buf.extend_from_slice(b"FRACTAL-CAV-TRAJ-v1");
    buf.extend_from_slice(&(n as u64).to_le_bytes());
    buf.extend_from_slice(m);
    sha256(&buf)
}

/// Raíz de la trayectoria: SHA-256(etiqueta ‖ n ‖ Merkle de los `state_root` 0..n-1). El número
/// de estados va dentro: sin él, la misma raíz valdría para árboles de otro tamaño.
pub fn trajectory_root(states: &[H]) -> H {
    let leaves: Vec<H> = states
        .iter()
        .map(|s| {
            let mut buf = [0u8; 33];
            buf[1..].copy_from_slice(s);
            sha256(&buf)
        })
        .collect();
    sellar_n(states.len(), &merkle(leaves))
}

fn node(l: &H, r: &H) -> H {
    let mut buf = [0u8; 65];
    buf[0] = 0x01;
    buf[1..33].copy_from_slice(l);
    buf[33..65].copy_from_slice(r);
    sha256(&buf)
}
fn traj_leaf(s: &H) -> H {
    let mut buf = [0u8; 33];
    buf[1..].copy_from_slice(s);
    sha256(&buf)
}

/// Camino Merkle del `state_root` del paso `idx` dentro de la trayectoria. Cada elemento es
/// `Some(hermano)` o `None` cuando el nodo subió sin pareja.
pub fn trajectory_path(states: &[H], mut idx: usize) -> Vec<Option<H>> {
    let mut level: Vec<H> = states.iter().map(traj_leaf).collect();
    let mut path = Vec::new();
    while level.len() > 1 {
        let sib = idx ^ 1;
        path.push(if sib < level.len() { Some(level[sib]) } else { None });
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for pair in level.chunks(2) {
            next.push(if pair.len() == 2 { node(&pair[0], &pair[1]) } else { pair[0] });
        }
        level = next;
        idx /= 2;
    }
    path
}

/// ¿`state` es el paso `idx` de una trayectoria de `n` estados con raíz `root`?
/// La forma del camino (dónde hay hermano y dónde no) la fija `n`: no la elige quien prueba.
pub fn trajectory_verify(state: &H, mut idx: usize, n: usize, path: &[Option<H>], root: &H) -> bool {
    if idx >= n {
        return false;
    }
    let mut h = traj_leaf(state);
    let mut width = n;
    let mut used = 0;
    while width > 1 {
        let Some(step) = path.get(used) else { return false };
        let sib = idx ^ 1;
        match (sib < width, step) {
            (true, Some(s)) => h = if idx % 2 == 0 { node(&h, s) } else { node(s, &h) },
            (false, None) => {}
            _ => return false,
        }
        used += 1;
        idx /= 2;
        width = (width + 1) / 2;
    }
    used == path.len() && sellar_n(n, &h) == *root
}
