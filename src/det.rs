//! Transcendentales propias en f32. Solo usan + - * / IEEE-754 (correctamente redondeadas en
//! toda arquitectura) y manipulación de bits: NADA de libm del sistema, NADA de FMA.
//! Son las MISMAS funciones que midió la sonda F1 (variante b) en Apple M5 y AMD EPYC.
//! Cambiar una constante aquí cambia todos los hashes: hay que subir `model::ENGINE_VERSION`.
//!
//! Precisión: en el rango que usa el modelo, el error relativo es ~1e-7. FUERA de él no sirven
//! como funciones de propósito general (tanh pierde precisión relativa con |x| < 1e-6, sqrt y ln
//! fallan con subnormales). No reutilizarlas sin acotar el dominio.

#[inline(never)]
pub fn exp_det(x: f32) -> f32 {
    if x < -87.0 {
        return 0.0;
    }
    if x > 88.0 {
        return f32::INFINITY;
    }
    let kf = x * 1.442_695_f32;
    let k = if kf >= 0.0 { (kf + 0.5) as i32 } else { (kf - 0.5) as i32 };
    let kk = k as f32;
    // Cody-Waite: ln2 = HI + LO, HI con mantisa corta para que kk*HI sea exacto.
    let r = (x - kk * f32::from_bits(0x3f31_7200)) - kk * f32::from_bits(0x35bf_be8e);
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    p * f32::from_bits(((k + 127) as u32) << 23)
}

#[inline(never)]
pub fn ln_det(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if m > 1.414_213_5 {
        m = m * 0.5;
        e += 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let s = z
        * (2.0
            + z2 * (2.0 / 3.0 + z2 * (2.0 / 5.0 + z2 * (2.0 / 7.0 + z2 * (2.0 / 9.0 + z2 * (2.0 / 11.0))))));
    e as f32 * 0.693_147_2 + s
}

#[inline(never)]
pub fn sqrt_det(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = f32::from_bits(0x1fbd_1df5u32.wrapping_add(x.to_bits() >> 1));
    for _ in 0..4 {
        y = 0.5 * (y + x / y);
    }
    y
}

#[inline(never)]
pub fn tanh_det(x: f32) -> f32 {
    let a = if x < 0.0 { -x } else { x };
    let t = if a > 9.0 {
        1.0
    } else {
        let e = exp_det(-(a + a));
        (1.0 - e) / (1.0 + e)
    };
    if x < 0.0 {
        -t
    } else {
        t
    }
}

/// Canario: ¿el compilador fusionó `a*b + c` en una FMA? Con estos valores, sin fusión el
/// resultado es exactamente 0 (a*b redondea a 1+2^-11); con FMA sale 2^-24.
/// Si esto devuelve `false`, NINGÚN hash de este binario es comparable con otro.
pub fn sin_fma() -> bool {
    let a = std::hint::black_box(1.0f32 + 1.0 / 4096.0);
    let b = std::hint::black_box(1.0f32 + 1.0 / 4096.0);
    let c = std::hint::black_box(-(1.0f32 + 1.0 / 2048.0));
    a * b + c == 0.0
}

/// Canario de subnormales: con «flush to zero» o «denormals are zero» activos en el hilo (basta
/// que el proceso anfitrión haya enlazado algo compilado con fast-math), esto da 0.
pub fn con_subnormales() -> bool {
    let x = std::hint::black_box(f32::MIN_POSITIVE);
    let h = std::hint::black_box(0.5f32);
    x * h != 0.0 && (x * h) + (x * h) == x
}

/// Prueba de humo del entorno numérico del hilo actual. NO es la garantía: la garantía es que
/// rustc no contrae `a*b+c` ni reasocia sin fast-math. Esto detecta un entorno roto evidente.
pub fn entorno_ok() -> bool {
    sin_fma() && con_subnormales()
}

/// Producto escalar con ORDEN DE REDUCCIÓN FIJO.
///
/// - `lanes8 == false`: acumulación secuencial `acc = init; acc += x[i]*w[i]` (la de la sonda).
/// - `lanes8 == true`: ocho acumuladores; el carril `j` suma los índices `i ≡ j (mod 8)` en
///   orden creciente sobre los bloques completos de 8; después
///   `init + (((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)))` y por último la cola, secuencial.
///   El orden está fijado por esta definición, no por el compilador: vectorice o no
///   (NEON, SSE2, AVX), el resultado es el mismo bit a bit.
#[inline(always)]
pub fn dot(x: &[f32], w: &[f32], init: f32, lanes8: bool) -> f32 {
    debug_assert_eq!(x.len(), w.len());
    if !lanes8 {
        let mut acc = init;
        for i in 0..x.len() {
            acc += x[i] * w[i];
        }
        return acc;
    }
    let mut a = [0.0f32; 8];
    let xc = x.chunks_exact(8);
    let wc = w.chunks_exact(8);
    let (xr, wr) = (xc.remainder(), wc.remainder());
    for (xs, ws) in xc.zip(wc) {
        for j in 0..8 {
            a[j] += xs[j] * ws[j];
        }
    }
    let mut acc = init + (((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7])));
    for i in 0..xr.len() {
        acc += xr[i] * wr[i];
    }
    acc
}
