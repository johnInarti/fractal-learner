//! Pruebas del registro de épocas (F3). Una por hallazgo de la segunda auditoría.
use fractal_learner::epoca::{correr, partir, repetir, rutas, Trabajo};
use fractal_learner::model::{Cfg, Model};
use std::fs;
use std::path::PathBuf;

const CORPUS: &[u8] = include_bytes!("../data/dominio-publico.txt");
const HOLD: usize = 39_244;

fn t(hilos: usize) -> Trabajo<'static> {
    Trabajo { preset: "mini", cfg: Cfg::mini(), corpus: CORPUS, holdout: HOLD, pasos: 3, eval_lotes: 2, hilos }
}
fn dir(nombre: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fl-epoca-{}-{nombre}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}
fn registro(d: &PathBuf) -> String {
    fs::read_to_string(rutas(d).registro).unwrap()
}

#[test]
fn dos_directorios_iguales_y_un_tercero_lo_repite() {
    let (a, b) = (dir("a"), dir("b"));
    for _ in 0..3 {
        correr(&t(1), &a).unwrap();
        correr(&t(3), &b).unwrap();
    }
    let cuerpos = |d: &PathBuf| registro(d).lines().map(|l| l[..l.find(",\"medido\"").unwrap()].to_string()).collect::<Vec<_>>();
    assert_eq!(cuerpos(&a), cuerpos(&b), "1 y 3 hilos dan los mismos cuerpos");
    assert_eq!(fs::read(rutas(&a).actual).ok(), fs::read(rutas(&b).actual).ok());
    let (n, _) = repetir(&t(2), &registro(&a)).unwrap();
    assert_eq!(n, 3);
}

#[test]
fn una_linea_antigua_alterada_se_detecta() {
    let d = dir("m1");
    correr(&t(1), &d).unwrap();
    correr(&t(1), &d).unwrap();
    let r = registro(&d);
    fs::write(rutas(&d).registro, r.replacen("\"pasos\":3", "\"pasos\":4", 1)).unwrap();
    assert!(correr(&t(1), &d).unwrap_err().contains("no es el hash"));
    // un registro inventado de una línea tampoco entra
    fs::write(rutas(&d).registro, "{\"cuerpo_sha256\":\"ff\",\"model_root_vigente\":\"00\"}\n").unwrap();
    assert!(correr(&t(1), &d).is_err());
    assert!(repetir(&t(1), &r.replacen("\"pasos\":3", "\"pasos\":4", 1)).is_err());
}

#[test]
fn la_epoca_cero_parte_del_modelo_de_la_semilla() {
    let d = dir("m2");
    fs::create_dir_all(&d).unwrap();
    let mut otro = Model::new(Cfg::mini(), 1);
    otro.p.wte[0] += 0.5;
    fs::write(rutas(&d).actual, otro.save_ckpt()).unwrap();
    assert!(correr(&t(1), &d).is_err(), "un checkpoint ajeno sin registro no es el génesis");
}

#[test]
fn un_corte_antes_de_renombrar_se_recupera_solo() {
    let d = dir("m3");
    correr(&t(1), &d).unwrap();
    let antes = fs::read(rutas(&d).actual).unwrap();
    let reg_antes = registro(&d);
    let l = correr(&t(1), &d).unwrap();
    assert!(l.contains("\"adoptada\":true"));
    // simular el corte: la línea quedó escrita, el candidato no llegó a renombrarse
    fs::rename(rutas(&d).actual, rutas(&d).candidato).unwrap();
    fs::write(rutas(&d).actual, &antes).unwrap();
    correr(&t(1), &d).unwrap();
    assert!(!rutas(&d).candidato.exists(), "el candidato pendiente se renombró al recuperarse");
    assert_eq!(registro(&d).lines().count(), 3);
    assert!(repetir(&t(1), &registro(&d)).is_ok());
    // corte ANTES de escribir la línea: el candidato sobra y el vigente sigue siendo el de antes
    let d2 = dir("m3b");
    correr(&t(1), &d2).unwrap();
    fs::write(rutas(&d2).candidato, b"basura").unwrap();
    correr(&t(1), &d2).unwrap();
    assert_eq!(reg_antes.lines().count(), 1);
    assert_eq!(registro(&d2).lines().count(), 2, "un candidato sobrante no rompe nada");
    assert!(repetir(&t(1), &registro(&d2)).is_ok());
}

#[test]
fn una_linea_a_medias_al_final_se_descarta() {
    let d = dir("m4");
    correr(&t(1), &d).unwrap();
    let r = registro(&d);
    fs::write(rutas(&d).registro, format!("{r}{{\"cuerpo\":{{\"formato\":\"FRACTAL-CAV-EP")).unwrap();
    let l = correr(&t(1), &d).unwrap();
    assert!(l.contains("\"epoca\":1,"), "la época siguiente es la 1, no se pega a la línea rota");
    assert_eq!(registro(&d).lines().count(), 2);
    assert!(repetir(&t(1), &registro(&d)).is_ok());
}

#[test]
fn dos_epocas_a_la_vez_no() {
    let d = dir("m5");
    correr(&t(1), &d).unwrap();
    let f = fs::OpenOptions::new().write(true).open(rutas(&d).cerrojo).unwrap();
    f.lock().unwrap();
    assert!(correr(&t(1), &d).unwrap_err().contains("otra época"));
    drop(f);
    correr(&t(1), &d).unwrap();
}

#[test]
fn el_control_no_puede_cambiar() {
    let d = dir("m6");
    correr(&t(1), &d).unwrap();
    let mut otro = t(1);
    otro.holdout = 5000;
    assert!(correr(&otro, &d).unwrap_err().contains("control"));
}

#[test]
fn parametros_hostiles() {
    let d = dir("hostil");
    let mut x = t(1);
    x.holdout = usize::MAX;
    assert!(correr(&x, &d).is_err());
    let mut x = t(1);
    x.eval_lotes = 1 << 62;
    assert!(correr(&x, &d).is_err());
    let mut x = t(1);
    x.pasos = 10_000_000;
    assert!(correr(&x, &d).unwrap_err().contains("entre 1 y"));
}

/// Tercera auditoría, M-2: una clave repetida o fuera de orden no pasa, aunque el hash cuadre.
#[test]
fn plantilla_estricta() {
    use fractal_learner::sha256::{hex, sha256};
    let d = dir("plantilla");
    correr(&t(1), &d).unwrap();
    let l = registro(&d);
    let l = l.trim_end();
    assert!(partir(l).is_some());
    let cuerpo = &l[10..l.find(",\"cuerpo_sha256\"").unwrap()];
    let falso = cuerpo.replacen("{\"formato\"", "{\"model_root_vigente\":\"ff\",\"formato\"", 1);
    let linea = format!("{{\"cuerpo\":{falso},\"cuerpo_sha256\":\"{}\",\"medido\":{{}}}}", hex(&sha256(falso.as_bytes())));
    assert!(partir(&linea).is_none(), "clave de más delante");
    fs::write(rutas(&d).registro, format!("{linea}\n")).unwrap();
    assert!(correr(&t(1), &d).is_err());
    assert!(partir(&l.replacen("\"medido\":{", "\"medido\":{\"x\":{", 1)).is_none(), "llaves dentro de medido");
    assert!(partir(&format!("{l} ")).is_none(), "basura al final");
}

/// M-3: con otro corpus el registro dejaría de poder repetirse: se rechaza al correr.
#[test]
fn otro_corpus_se_rechaza() {
    let d = dir("corpus");
    correr(&t(1), &d).unwrap();
    let mut c = CORPUS.to_vec();
    c[0] ^= 1;
    let otro = Trabajo { corpus: Box::leak(c.into_boxed_slice()), ..t(1) };
    assert!(correr(&otro, &d).unwrap_err().contains("otro corpus"));
}

/// M-4: una última línea entera sin el salto final se conserva, no se trunca; una línea a medias
/// se aparta a un fichero (no se borra).
#[test]
fn cola_entera_se_conserva_y_la_rota_se_aparta() {
    let d = dir("cola");
    correr(&t(1), &d).unwrap();
    correr(&t(1), &d).unwrap();
    let r = registro(&d);
    fs::write(rutas(&d).registro, r.trim_end()).unwrap();
    let l = correr(&t(1), &d).unwrap();
    assert!(l.contains("\"epoca\":2,"), "la época 1 seguía comprometida");
    let r = registro(&d);
    fs::write(rutas(&d).registro, format!("{r}{{\"cuerpo\":{{\"form")).unwrap();
    correr(&t(1), &d).unwrap();
    assert_eq!(registro(&d).lines().count(), 4);
    let apartadas: Vec<_> = fs::read_dir(&d).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().starts_with("epocas.cola-")).collect();
    assert_eq!(apartadas.len(), 1, "la cola rota quedó guardada aparte");
    assert!(repetir(&t(1), &registro(&d)).is_ok());
}

/// Corte en la PRIMERA adopción: candidato a medias y sin actual.ckpt. Se trata como ausente.
#[test]
fn candidato_a_medias_en_la_primera_epoca() {
    let d = dir("primera");
    fs::create_dir_all(&d).unwrap();
    fs::write(rutas(&d).candidato, b"FLCK1\0\0\0a medias").unwrap();
    correr(&t(1), &d).unwrap();
    assert_eq!(registro(&d).lines().count(), 1);
}
