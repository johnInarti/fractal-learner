# fractal-learner

Motor de entrenamiento **determinista** en CPU, en Rust y sin dependencias: el mismo entrenamiento da la misma
trayectoria byte a byte en cualquier máquina y con cualquier número de hilos. Es la fase F1 de la Cadena de
Aprendizaje Verificable de FractalAI (en diseño; no está construida). Estado: pre-piloto.

## Compruébalo tú (≈3 min con la compilación, rustc 1.97.1)

```
cargo build --release && ./target/release/fractal-learner train --preset athena --corpus data/dominio-publico.txt \
  --holdout 30000 --eval-batches 2 --steps 30 --threads 4 --out runs/athena && \
cmp runs/athena/trayectoria.tsv referencia-dominio-publico/athena/trayectoria.tsv
```

Si `cmp` no imprime nada, tu trayectoria es idéntica a la nuestra. Si difiere, abre un issue con tu CPU, sistema
operativo, versión de rustc y la salida de `cmp`: nos interesa sobre todo cuando NO coincide.

El corpus es de dominio público: los primeros 299 897 bytes del Quijote (Project Gutenberg #2000); procedencia y
SHA-256 en `data/DOMINIO-PUBLICO.md`.

## Estado del arte

No inventamos la técnica. El entrenamiento reproducible bit a bit está publicado antes y a mayor escala: Gensyn,
Verde (arXiv:2502.19405) y OPEN-1B (arXiv:2609.17380); Microsoft, RepDL (arXiv:2510.09180). Esto es un motor mínimo
que cualquiera compila y verifica sin infraestructura especial.

## Qué garantiza y cómo

| Regla | Dónde |
|---|---|
| Nada de libm del sistema | `src/det.rs`: exp, ln, sqrt y tanh propias, solo con `+ - * /` IEEE |
| Nada de FMA ni fast-math | rustc no contrae `a*b+c` ni reasocia: esa es la garantía. `det::entorno_ok()` es una prueba de humo (FMA y subnormales a cero) que corre en cada paso |
| Un resultado no finito no tiene raíz | Los bits de un NaN dependen de la arquitectura y del compilador: `Model::step` devuelve error, igual para todos |
| Cada raíz dice con qué motor se hizo | `ENGINE_VERSION` y el modo del producto escalar entran en `model_root`, `state_root` y `job_id` |
| Reducciones en orden fijo | `det::dot`: secuencial, o 8 carriles con árbol de suma fijado por el código |
| Los hilos no cambian nada | `src/par.rs`: cada salida tiene un solo hilo dueño; ningún acumulador se comparte |

## Qué compromete un entrenamiento

- `model_root`: raíz por contenido de los pesos (Merkle por tensor, hojas de 2^18 f32).
- `state_root`: pesos + momentos de AdamW + contador. Es lo que hace falta para re-ejecutar el paso siguiente.
- `trayectoria_root`: SHA-256(etiqueta ‖ número de estados ‖ Merkle de los `state_root` de los pasos 0..=N). El entrenador la publica ANTES
  de saber qué pasos se le van a revisar (el muestreo Fiat-Shamir es de F2).
- `job_id`: configuración + corpus + partición de control + pasos. El número de hilos no entra.

`trayectoria.tsv` y `compromiso.txt` son deterministas y se comparan byte a byte entre máquinas.
`resumen.txt` y `tiempos.tsv` llevan lo que sí depende de la máquina.

## Verificar un paso

```
fractal-learner verify --preset P --ckpt ckpt_K.bin --corpus C --holdout H --from <state_root K> --expect <state_root K+1>
```

Las DOS raíces son obligatorias. Comprobar solo la de llegada no basta: un bit bajo cambiado en un
momento de Adam puede absorberse por redondeo y dar el mismo estado siguiente (visto una vez al
escribir las pruebas; la prueba actual comprueba que la raíz de partida siempre cambia).

La configuración la pone quien verifica (`--preset`), nunca el checkpoint: un checkpoint con otra
tasa de aprendizaje, otro lote u otra versión del motor se rechaza.

## Presets

| Preset | Forma | Parámetros | Uso |
|---|---|---|---|
| `sonda` | d=64, 2 capas, producto secuencial | 118 K | Reproduce bit a bit la sonda F1 (`compat_sonda`) |
| `mini` | d=96, 2 capas, 8 carriles | 252 K | Pruebas |
| `athena` | d=512, 4 capas, 4 cabezas, V=16 384, T=64, B=8 | 21,0 M | La forma de ATHENA en producción |

Los datos de esta fase son bytes. El preset `athena` tiene los tensores y el coste reales, no el
tokenizador BPE ni el corpus de producción: eso es trabajo de datos, posterior.

## Límites, dichos tal cual

- Un cambio en `det.rs`, en el orden de un bucle o en una constante cambia todos los hashes: es una
  versión nueva del motor y hay que regenerar las referencias.
- El determinismo está medido en Apple M5 Pro, Intel Xeon, AMD EPYC y ARM Neoverse (runs 43940091602 y
  43954504503 del repositorio de origen). No con otros compiladores que rustc 1.97.1.
- CPU solamente. No hay GPU en este motor.

## Puerta F1 sobre el motor real

La trayectoria de la forma de ATHENA (21,0 M de parámetros, 30 pasos) y la de `mini` (300 pasos) se
comparan byte a byte con `referencia-dominio-publico/` en Intel x86 y ARM Neoverse en cada push del repositorio
de origen, junto con la re-ejecución de un paso contra las raíces de la otra arquitectura. Las referencias cambian cuando cambia el formato de las
raíces; la última confirmación en x86 real se anota en `F2-RESULTADOS.md`.

No medido: tokenizador BPE y corpus de producción, más de 30 pasos a la forma de ATHENA, otras
versiones de rustc que la 1.97.1, GPU.
