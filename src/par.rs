//! Paralelismo que NO puede cambiar un resultado: cada elemento de salida tiene UN solo hilo
//! dueño, que lo calcula entero con su bucle interno en orden fijo. Los hilos nunca suman
//! sobre el mismo acumulador, así que el número de hilos no entra en ningún hash.
use std::ops::Range;

/// Reparte `rows` filas en bloques contiguos, uno por hilo, y entrega a `f` el rango de filas
/// y el trozo correspondiente de cada salida. Una salida vacía tiene filas de tamaño 0.
pub fn split3<F>(threads: usize, rows: usize, a: &mut [f32], b: &mut [f32], c: &mut [f32], f: F)
where
    F: Fn(Range<usize>, &mut [f32], &mut [f32], &mut [f32]) + Sync,
{
    if rows == 0 {
        return;
    }
    let (ra, rb, rc) = (a.len() / rows, b.len() / rows, c.len() / rows);
    assert!(a.len() == ra * rows && b.len() == rb * rows && c.len() == rc * rows, "salida no múltiplo de filas");
    let th = threads.max(1).min(rows);
    if th == 1 {
        f(0..rows, a, b, c);
        return;
    }
    std::thread::scope(|s| {
        let (mut a, mut b, mut c) = (a, b, c);
        let mut start = 0usize;
        for k in 0..th {
            let end = rows * (k + 1) / th;
            let cnt = end - start;
            let (a0, a1) = std::mem::take(&mut a).split_at_mut(cnt * ra);
            let (b0, b1) = std::mem::take(&mut b).split_at_mut(cnt * rb);
            let (c0, c1) = std::mem::take(&mut c).split_at_mut(cnt * rc);
            a = a1;
            b = b1;
            c = c1;
            let fr = &f;
            let r = start..end;
            if k == th - 1 {
                fr(r, a0, b0, c0);
            } else {
                s.spawn(move || fr(r, a0, b0, c0));
            }
            start = end;
        }
    });
}

pub fn split1<F>(threads: usize, rows: usize, a: &mut [f32], f: F)
where
    F: Fn(Range<usize>, &mut [f32]) + Sync,
{
    split3(threads, rows, a, &mut [], &mut [], |r, x, _, _| f(r, x));
}

/// Igual que `split1` pero para salidas de 32 bytes (hojas de Merkle).
pub fn split_hash<F>(threads: usize, out: &mut [[u8; 32]], f: F)
where
    F: Fn(Range<usize>, &mut [[u8; 32]]) + Sync,
{
    let rows = out.len();
    if rows == 0 {
        return;
    }
    let th = threads.max(1).min(rows);
    if th == 1 {
        f(0..rows, out);
        return;
    }
    std::thread::scope(|s| {
        let mut rest = out;
        let mut start = 0usize;
        for k in 0..th {
            let end = rows * (k + 1) / th;
            let (o0, o1) = std::mem::take(&mut rest).split_at_mut(end - start);
            rest = o1;
            let fr = &f;
            let r = start..end;
            if k == th - 1 {
                fr(r, o0);
            } else {
                s.spawn(move || fr(r, o0));
            }
            start = end;
        }
    });
}
