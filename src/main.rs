use fractal_learner::model::{Cfg, Model};
use fractal_learner::sha256::{hex, sha256};
use fractal_learner::{det, reexecute_step, train, Job};
use std::io::Write;
use std::time::Instant;

fn arch() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else {
        "otra"
    }
}

fn flag<'a>(a: &'a [String], name: &str) -> Option<&'a str> {
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).map(|s| s.as_str())
}
fn num<T: std::str::FromStr>(a: &[String], name: &str, def: T) -> T {
    match flag(a, name) {
        Some(s) => s.parse().unwrap_or_else(|_| fail(&format!("valor inválido para {name}: {s}"))),
        None => def,
    }
}
fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2)
}

const USO: &str = "uso:
  fractal-learner --version
  fractal-learner selftest
  fractal-learner train  --preset sonda|mini|athena --corpus F --steps N --out DIR
                         [--holdout BYTES] [--eval-batches K] [--threads T] [--ckpt 0,5,10]
  fractal-learner epoca  --preset P --corpus F --holdout BYTES --dir DIR --pasos H [--eval-batches K] [--threads T]
  fractal-learner repetir-epocas --preset P --corpus F --holdout BYTES --registro epocas.jsonl [--threads T]
  fractal-learner verify --preset P --ckpt F --corpus F --holdout BYTES --from STATE_ROOT_K --expect STATE_ROOT_K+1 [--threads T]";

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("selftest") => selftest(),
        Some("--version") | Some("version") => println!(
            "fractal-learner {} motor={} arq={}",
            env!("CARGO_PKG_VERSION"), fractal_learner::model::ENGINE_VERSION, arch()
        ),
        Some("train") => cmd_train(&a),
        Some("verify") => cmd_verify(&a),
        Some("epoca") => cmd_epoca(&a),
        Some("repetir-epocas") => cmd_repetir(&a),
        _ => fail(USO),
    }
}

fn selftest() {
    assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    println!("sha256 OK");
    println!("arq={} sin_fma={} con_subnormales={}", arch(), det::sin_fma(), det::con_subnormales());
    if !det::entorno_ok() {
        fail("entorno numérico NO apto");
    }
}

fn cmd_train(a: &[String]) {
    let preset = flag(a, "--preset").unwrap_or_else(|| fail(USO));
    let cfg = Cfg::preset(preset).unwrap_or_else(|| fail("preset desconocido"));
    let corpus_path = flag(a, "--corpus").unwrap_or_else(|| fail(USO));
    let out = flag(a, "--out").unwrap_or_else(|| fail(USO)).to_string();
    let steps: u64 = num(a, "--steps", 0);
    let holdout: usize = num(a, "--holdout", 0);
    let eval_batches: usize = num(a, "--eval-batches", 8);
    let threads: usize = num(a, "--threads", 1);
    let ckpts: Vec<u64> = flag(a, "--ckpt")
        .map(|s| s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap_or_else(|_| fail("--ckpt inválido"))).collect())
        .unwrap_or_default();
    let corpus = std::fs::read(corpus_path).unwrap_or_else(|e| fail(&format!("no se pudo leer el corpus: {e}")));
    if holdout + cfg.t + 2 >= corpus.len() {
        fail("holdout demasiado grande para el corpus");
    }
    if steps == 0 {
        fail("--steps debe ser > 0");
    }
    std::fs::create_dir_all(&out).unwrap();
    let job = Job { cfg: cfg.clone(), corpus: &corpus, holdout, steps, eval_batches };
    let mut tiempos = std::fs::File::create(format!("{out}/tiempos.tsv")).unwrap();
    writeln!(tiempos, "paso\tms_desde_el_anterior").unwrap();
    let t0 = Instant::now();
    let mut last = Instant::now();
    let mut params = 0usize;
    let com = train(&job, threads, |s, m: &Model| {
        params = m.p.count();
        writeln!(tiempos, "{s}\t{:.3}", last.elapsed().as_secs_f64() * 1e3).unwrap();
        last = Instant::now();
        if ckpts.contains(&s) {
            std::fs::write(format!("{out}/ckpt_{s}.bin"), m.save_ckpt()).unwrap();
        }
    });
    let total = t0.elapsed().as_secs_f64();

    // trayectoria.tsv y compromiso.txt son DETERMINISTAS: se comparan byte a byte entre máquinas.
    let mut tr = String::from("paso\tstate_root\tmodel_root\tperdida_lote_bits\n");
    for s in 0..=steps as usize {
        let lb = if s == 0 { "-".to_string() } else { format!("{:016x}", com.loss_bits[s - 1]) };
        tr.push_str(&format!("{s}\t{}\t{}\t{lb}\n", hex(&com.states[s]), hex(&com.models[s])));
    }
    std::fs::write(format!("{out}/trayectoria.tsv"), &tr).unwrap();
    let ev = |b: Option<u64>| b.map(|x| format!("{:016x}", x)).unwrap_or_else(|| "-".into());
    let compromiso = format!(
        "formato=FRACTAL-CAV-COMPROMISO-v1\njob_id={}\npreset={preset}\ncfg={}\ncorpus_sha256={}\ncorpus_bytes={}\nholdout_bytes={holdout}\npasos={steps}\neval_lotes={eval_batches}\nparametros={params}\ntrayectoria_root={}\nstate_root_final={}\nmodel_root_final={}\neval_control_antes_bits={}\neval_control_despues_bits={}\n",
        hex(&com.job_id), hex(&cfg.bytes()), hex(&sha256(&corpus)), corpus.len(),
        hex(&com.trajectory_root), hex(&com.states[steps as usize]), hex(&com.models[steps as usize]),
        ev(com.eval_before_bits), ev(com.eval_after_bits),
    );
    std::fs::write(format!("{out}/compromiso.txt"), &compromiso).unwrap();

    let f = |b: Option<u64>| b.map(|x| format!("{:.6}", f64::from_bits(x))).unwrap_or_else(|| "-".into());
    let resumen = format!(
        "arq={}\nhilos={threads}\nsegundos_total={total:.3}\nms_por_paso_medio={:.3}\nperdida_lote_primera={:.6}\nperdida_lote_ultima={:.6}\neval_control_antes={}\neval_control_despues={}\n",
        arch(), total * 1e3 / steps as f64,
        f64::from_bits(com.loss_bits[0]), f64::from_bits(com.loss_bits[steps as usize - 1]),
        f(com.eval_before_bits), f(com.eval_after_bits),
    );
    std::fs::write(format!("{out}/resumen.txt"), &resumen).unwrap();
    print!("{compromiso}{resumen}");
}

fn cmd_verify(a: &[String]) {
    let ckpt_path = flag(a, "--ckpt").unwrap_or_else(|| fail(USO));
    let corpus_path = flag(a, "--corpus").unwrap_or_else(|| fail(USO));
    let expect = flag(a, "--expect").unwrap_or_else(|| fail(USO));
    // Las DOS raíces son obligatorias: comprobar solo la de llegada deja pasar checkpoints
    // alterados cuyo cambio se absorbe por redondeo.
    let from = flag(a, "--from").unwrap_or_else(|| fail(USO));
    // La configuración la pone QUIEN VERIFICA, no el checkpoint.
    let cfg = Cfg::preset(flag(a, "--preset").unwrap_or_else(|| fail(USO))).unwrap_or_else(|| fail("preset desconocido"));
    let holdout: usize = num(a, "--holdout", 0);
    let threads: usize = num(a, "--threads", 1);
    let corpus = std::fs::read(corpus_path).unwrap_or_else(|e| fail(&format!("corpus: {e}")));
    let ckpt = std::fs::read(ckpt_path).unwrap_or_else(|e| fail(&format!("checkpoint: {e}")));
    if holdout >= corpus.len() {
        fail("holdout mayor que el corpus");
    }
    let t0 = Instant::now();
    let (s, before, roots) = reexecute_step(&ckpt, &cfg, &corpus[..corpus.len() - holdout], cfg.seed, threads).unwrap_or_else(|e| fail(&e));
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let got = hex(&roots.state);
    let ok_from = hex(&before) == from;
    let ok = ok_from && got == expect;
    println!(
        "arq={} hilos={threads} paso_reejecutado={s} partida={} PARTIDA_COINCIDE={ok_from} llegada={got} esperado={expect} COINCIDE={ok} ms_total={ms:.1} bytes_ckpt={}",
        arch(), hex(&before), ckpt.len()
    );
    if !ok {
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------------- épocas (F3)
fn trabajo_de<'a>(a: &'a [String], corpus: &'a [u8]) -> fractal_learner::epoca::Trabajo<'a> {
    let preset = flag(a, "--preset").unwrap_or_else(|| fail(USO));
    fractal_learner::epoca::Trabajo {
        preset,
        cfg: Cfg::preset(preset).unwrap_or_else(|| fail("preset desconocido")),
        corpus,
        holdout: num(a, "--holdout", 0),
        pasos: num(a, "--pasos", 0),
        eval_lotes: num(a, "--eval-batches", 16),
        hilos: num(a, "--threads", 1),
    }
}

fn cmd_epoca(a: &[String]) {
    let corpus = std::fs::read(flag(a, "--corpus").unwrap_or_else(|| fail(USO))).unwrap_or_else(|e| fail(&format!("corpus: {e}")));
    let t = trabajo_de(a, &corpus);
    let dir = flag(a, "--dir").unwrap_or_else(|| fail(USO));
    match fractal_learner::epoca::correr(&t, std::path::Path::new(dir)) {
        Ok(l) => print!("{l}"),
        Err(e) => fail(&e),
    }
}

fn cmd_repetir(a: &[String]) {
    let corpus = std::fs::read(flag(a, "--corpus").unwrap_or_else(|| fail(USO))).unwrap_or_else(|e| fail(&format!("corpus: {e}")));
    let mut a2 = a.to_vec();
    a2.extend(["--pasos".into(), "1".into()]); // cada época trae sus pasos en el registro
    let t = trabajo_de(&a2, &corpus);
    let reg = std::fs::read_to_string(flag(a, "--registro").unwrap_or_else(|| fail(USO))).unwrap_or_else(|e| fail(&format!("registro: {e}")));
    match fractal_learner::epoca::repetir(&t, &reg) {
        Ok((n, vig)) => println!("REPETIDO: {n} épocas idénticas byte a byte desde la semilla; model_root_vigente={vig}"),
        Err(e) => fail(&e),
    }
}
