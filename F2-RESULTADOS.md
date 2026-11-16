# F2: simulación del protocolo sobre el motor real (versión 2, 2026-11-16)

> **Copia pública:** estas cifras se midieron con un corpus interno de 334 KB que no se publica. Aquí `f2_sim`
> usa `data/dominio-publico.txt`, así que al repetirlo los números concretos pueden cambiar; el veredicto de la
> puerta F2 («no se cumple») no depende del corpus.

Reproducir: `cargo run --release --bin f2_sim` (unos 45 minutos en un Apple M5 Pro; `A` y `D` sueltos
tardan un minuto). Protocolo en `src/protocolo.rs`, pruebas en `tests/protocolo.rs`.

Esta versión sustituye a la primera, que no resistió su propia revisión. Agentes revisores
independientes la atacaron en tres vueltas (seguridad del protocolo, determinismo del motor, método
experimental, y después confirmación de cada arreglo); lo que encontraron está al final, en «Qué
cambió tras la revisión».

## Veredicto sobre la puerta de F2

La puerta pide: «el 100 % de los tramposos inyectados queda detectado o acotado».
**No se cumple.** Hay tramposos detectados siempre, tramposos que solo se frenan y uno que no se para.
La regla general que sale de la segunda auditoría: **todo tramposo puede reducirse a «un solo paso
falso»**, que el sorteo ve s/(h-1) de las veces; lo único que lo hace inofensivo es acotar lo que
puede mover (cota de desplazamiento + agregación robusta) y que la pena por ser visto supere la
ganancia.

| Tramposo | Resultado | Dato |
|---|---|---|
| No calcula nada, literalmente | Detectado siempre | 20 de 20 pasos falsos |
| No calcula casi nada: un estado inicial que es punto fijo (todo ceros) y 19 pasos «gratis» | Casi nunca en una ronda | Un solo paso falso: 10 % por ronda (segunda auditoría) |
| Copia la trayectoria entera de otra identidad | Detectado siempre | 20 de 20 pasos falsos: la semilla de lotes es por identidad |
| Entrena con otra configuración (otra tasa, otro lote) | Detectado siempre | El checkpoint con otra configuración se rechaza (prueba `otra_configuracion_no_verifica`) |
| Falsea el último paso para entregar pesos arbitrarios o ajenos | Detectado siempre | El último paso se re-ejecuta sin sorteo |
| Inventa el penúltimo estado y da honesto el último | Detectado a veces; pesos finales ACOTADOS | Un paso falso: 10 % por ronda. Sin cota elegía los pesos finales a 1 ulp de un objetivo cualquiera. Ahora `verificar_extremos` aplica la cota EXACTA de AdamW (`cota_desplazamiento`: con h=20, 1,06·h·lr por coordenada; un honesto llega a 1,04). Ese tramposo mueve menos que el envenenador «al tope» del experimento C (2,43·h·lr), que ya se midió. La primera cota (4·h·lr·(1+\|inicio\|)) lo dejaba 1,65 veces más fuerte que ese: lo encontró la tercera auditoría |
| Se salta 1 de cada 4 pasos | Detectado a veces | 45 % por ronda con el protocolo; 95 % sorteando 8 pasos |
| Un paso intermedio falso y sigue honesto | Casi nunca en una ronda | 10 % por ronda; 40 % sorteando 8 |
| Minoría envenenadora NO detectada (1 o 3 de 8) | **Frenada, sin cota** | Ver abajo: el daño crece con las rondas |
| Mayoría que envía copias iguales (5 de 8) | Frenada con un voto por cúmulo | +0,11 a +0,13 nats; sin eso, peor que sin entrenar |
| **Mayoría con envíos distintos entre sí (5 de 8)** | **No se para** | Pérdida de 128 a 1 820 con cualquier agregador probado (sin saber nada: 5,5) |

**La minoría no queda acotada.** Con mediana y sin detección, a 25 rondas un envenenador de ocho
cuesta de +0,02 a +0,03 nats, y tres cuestan de +0,14 a +0,24. A 60 rondas, uno cuesta +0,04 y
tres cuestan +0,37: con tres envenenadores, 60 rondas (3,29) no alcanzan lo que los honestos logran
en 25 (3,10). Cinco honestos solos llegan a 3,105; con los tres envenenadores dentro, la red de
ocho queda entre 3,24 y 3,35. Es decir, tenerlos dentro es peor que no tenerlos.

**La mayoría disfrazada no la arregla ningún agregador que cuente identidades**: si la mayoría de
los votos empuja en contra, la mediana es suya. Es un argumento, comprobado con los tres
agregadores probados. Lo que queda contra ella no se midió aquí:
- el coste de tener identidades (fianza), que es una decisión del fundador;
- la detección acumulada: para pasar el sorteo, el envenenador tiene que meter su veneno en un
  paso intermedio, y eso se detecta el 10 % de las rondas (40 % sorteando 8 pasos).

## Lo que F2 deja para el diseño

Lo que sale de una medición:
1. **Semilla de lotes por identidad y ronda.** El trabajo de una identidad no le sirve a otra: la
   copia de una trayectoria entera falla todos los pasos.
2. **El último paso se re-ejecuta siempre.** Sin esa regla, falsear solo el último paso entregaba
   pesos arbitrarios (o los de otro) y se detectaba el 10 % de las veces.
3. **La configuración la pone la ronda, no el checkpoint.** Sin esa comprobación, un tramposo
   entrenaba con su propia tasa de aprendizaje y pasaba 20 de 20 pasos.
4. **La media no sirve**: un solo envío grande la deja peor que sin entrenar.
5. **Conservar el optimizador entre rondas mejora 0,10 nats** (2,911 frente a 3,014). Obliga a que
   cada aprendiz comprometa su propio estado de una ronda a la siguiente; no está implementado en
   la verificación.
6. **«Un voto por cúmulo» solo para copias casi exactas** (coseno > 0,999 con todos los miembros).
   Para copias con parecido más flojo es peligroso: fundiría a los honestos cuando se parezcan, que
   es lo que pasa cuando el entrenamiento va bien. No es una defensa contra identidades falsas.
7. **La atribución «dejar uno fuera» no resiste a quien se duplica.** Con 2 copias el actor sube su
   cuota del 43 % al 56 % y cobra más en 23 de 40 sorteos. Con 3 cae al 8 %; con 8 es la mediana
   entera y nadie cobra. Además 2 de 5 honestos salen con recompensa negativa. Hace falta otra regla.

Lo que es propuesta, sin medir:
- **Que el verificador entregue una prueba que no sea pública.** La raíz de llegada ya la conoce todo
  el mundo, así que repetirla no prueba nada. El código da una raíz del gradiente con la identidad
  del verificador en cada hoja. No impide que dos verificadores coludidos, o el aprendiz y un
  verificador, se pasen el gradiente entero. Falta decidir contra qué se compara.
- **Un solo compromiso por identidad y ronda, fijado antes de la baliza**, y las identidades
  registradas antes de la baliza anterior. Sin eso el tramposo repite compromisos hasta que el
  sorteo le evita. Le toca a la cadena (F3).

## Calidad frente a centralizado

Con el mismo cómputo total, repartirlo entre 8 aprendices con 20 pasos locales cuesta **0,60 nats**
frente a hacerlo centralizado (3,014 frente a 2,41, tanto en serie como en paralelo con lote 32).
Conservando el optimizador, 0,50. Los 8 juntos mejoran 0,14 nats a un aprendiz solo con el mismo
protocolo (3,150), gastando ocho veces su cómputo. Sin tramposos, mediana, media y media recortada
no se distinguen.

## Límites

- Modelo de 252 K parámetros y datos por bytes: 295 KB de documentación del repo para entrenar y un
  documento aparte de 39 KB como control (el 1,4 % de sus tramos de 32 bytes aparece literal en el
  entrenamiento).
- Tres semillas de lotes por escenario en B y C, con los mismos pesos iniciales. Una sola ronda y
  una semilla en D, con 40 sorteos del control.
- Los costes son una sola medición y dependen de la carga de la máquina.
- Las copias de D son exactas. Un actor real partiría sus datos y enviaría deltas distintos: sin probar.
- La baliza es un hash local; la baliza externa es una dependencia pendiente.
- Todos entrenan con el mismo corpus comprometido. Con datos propios (F4), un aprendiz honrado en
  el cómputo puede entrenar con datos envenenados y la re-ejecución sale verde.

## Qué cambió tras la revisión

**Seguridad del protocolo** (demostrado con pruebas ejecutables por el auditor; cerrado en `77ecc5157`):
- El verificador usaba la configuración que venía dentro del checkpoint. Con tasa negativa, el
  tramposo pasaba extremos y 20 de 20 pasos entregando ascenso de gradiente.
- Con valores no numéricos, una trayectoria «válida» costaba 3 pasos reales de 20.
- Falsear el último paso entregaba pesos arbitrarios.
- Entradas hostiles tumbaban al verificador o le hacían reservar memoria sin límite.
- Los cúmulos por enlace simple dejaban que un «puente» quitara el voto a los honestos.

**Determinismo del motor** (cerrado en `35e44392d`):
- Un valor no numérico tiene bits distintos en x86 y en ARM y según las opciones del compilador:
  la misma trayectoria daba cinco raíces en cinco compilaciones. Ahora un paso no finito es un error.
- Las raíces no llevaban la versión del motor.
- El gradiente del modo de 8 carriles se comprobó por diferencias finitas contra una referencia en
  f64 escrita aparte: correcto. El reparto por hilos, sin defectos.

**Segunda y tercera vuelta** (cerrado en `e401a80b5` y siguientes):
- El penúltimo estado inventado entregaba pesos arbitrarios: cota de desplazamiento, después la exacta.
- El registro de épocas no se validaba al leer, el génesis no estaba atado, un corte lo atascaba,
  dos épocas podían correr a la vez, el control y el corpus podían cambiar: todo cerrado, con prueba.
- Dos lectores podían leer la misma línea de forma distinta (claves repetidas): plantilla estricta.
- La firma podía sellar una cabeza sobre un historial reescrito: ahora se niega.
- Las unidades usaban una lista negra de ficheros: ahora lista blanca, comprobada en un systemd real.

**Método** (esta versión):
- Tres semillas y rangos en vez de cifras sueltas; el control, cortado en frontera de documento.
- La comparación con centralizado usaba una sola semilla, que resultó favorable: la brecha era
  0,45 y es 0,60.
- «Acotado por la mediana» pasó a «frenado, sin cota»: no se había mirado el horizonte largo.
- «Dividirse no suma» era falso para 2 copias.
- El «verificador perezoso» eran dos comparaciones escritas a mano, no una medición: quitado.
- «El tope por coordenada no aporta nada» era trivial tal como se probó: quitado.

---

# Datos tal como los imprime el simulador

## A. Verificación por re-ejecución

Montaje: 251520 parámetros, 20 pasos locales, corpus de entrenamiento de 295400 bytes. Cada paso de cada envío se re-ejecutó de verdad con el verificador del protocolo para saber cuáles son falsos. «Sorteo» es la probabilidad exacta (hipergeométrica) de que `s` pasos sorteados toquen uno falso. «Protocolo» es lo medido con 20 000 balizas usando la regla completa: el último paso se re-ejecuta siempre, más 2 sorteados.

| Conducta | Extremos | Pasos falsos | Sorteo s=2 | Sorteo s=8 | Protocolo (último + 2) |
|---|---|---|---|---|---|
| honesto | válidos | 0 de 20 | 0.0 % | 0.0 % | 0.0 % |
| honesto (otra identidad) | válidos | 0 de 20 | 0.0 % | 0.0 % | 0.0 % |
| no calcula nada | válidos | 20 de 20 | 100.0 % | 100.0 % | 100.0 % |
| se salta 1 de cada 4 pasos, nunca el último | válidos | 5 de 20 | 44.7 % | 94.9 % | 45.4 % |
| un paso intermedio falso (el 10) y sigue honesto | válidos | 1 de 20 | 10.0 % | 40.0 % | 9.9 % |
| último paso falso: entrega pesos arbitrarios | válidos | 1 de 20 | 10.0 % | 40.0 % | 100.0 % |
| último paso falso: entrega los pesos de otro | válidos | 1 de 20 | 10.0 % | 40.0 % | 100.0 % |
| copia la trayectoria entera de otra identidad | válidos | 20 de 20 | 100.0 % | 100.0 % | 100.0 % |

**Coste en esta corrida** (una medición; varía con la carga de la máquina). Entrenar 20 pasos: 273 ms. Comprometer cada paso (raíces y checkpoints) añade 98 %. Re-ejecutar un paso con sus comprobaciones: 41.3 ms, 15.2 % del cómputo de la ronda; el protocolo «último + 2» cuesta 45 %. En un modelo tan pequeño manda el coste de los hashes.

## B. Calidad frente a centralizado (sin tramposos)

Montaje: 20 pasos locales por ronda, 40 rondas, lote 4, misma tasa de aprendizaje en todo. 3 semillas de lotes por fila: media [mínimo – máximo]. Pérdida en el control (39244 bytes: un documento entero que no está en el entrenamiento); al empezar es 5.461.

| Corrida | Muestras de entrenamiento | Pérdida en control |
|---|---|---|
| 8 aprendices, mediana | 25600 | 3.014 [3.012 – 3.018] |
| 8 aprendices, media | 25600 | 3.014 [3.007 – 3.019] |
| 8 aprendices, media recortada (quita 2 y 2) | 25600 | 3.025 [3.017 – 3.030] |
| 8 aprendices, mediana, conservando el optimizador entre rondas | 25600 | 2.911 [2.898 – 2.918] |
| 1 aprendiz con el mismo protocolo (reinicia el optimizador cada 20 pasos) | 3200 | 3.150 [3.125 – 3.189] |
| centralizado, 800 pasos, lote 4 (el cómputo de UN aprendiz) | 3200 | 3.111 [3.082 – 3.127] |
| centralizado, 800 pasos, lote 32 (el cómputo de los 8, en paralelo) | 25600 | 2.411 [2.300 – 2.587] |
| centralizado, 6400 pasos, lote 4 (el cómputo de los 8, en serie) | 25600 | 2.405 [2.360 – 2.466] |

El mayor desplazamiento de un peso en una ronda honesta fue 2.03 × (pasos × tasa de aprendizaje). El coseno entre los deltas de dos aprendices honestos fue 0.37 de media.

## C. Cuánto daña un envenenador que NO es detectado

Es el peor caso: los `k` envenenadores de 8 pasan la verificación en todas las rondas (25 rondas). Calculan su delta honesto y envían otro. «Al tope» = 2.43 × pasos × tasa de aprendizaje en cada coordenada, contra el signo de su propio delta honesto (no necesitan ver a los demás). «Cúmulo»: deltas con coseno > 0.999 con todos los miembros cuentan como un voto. Cada celda: pérdida final en el control, media [mínimo – máximo] de 3 semillas. Sin saber nada la pérdida es 5.545.

**Agregador: media.** Sin veneno: 3.091 [3.076 – 3.104].

| Ataque | k=1 | k=3 | k=5 |
|---|---|---|---|
| bruto: −10 × delta, todos iguales | 39.821 [33.576 – 47.090] peor que sin entrenar | 2338.812 [2091.774 – 2597.439] peor que sin entrenar | 9721.892 [9145.667 – 10531.104] peor que sin entrenar |
| al tope contra el signo, todos iguales | 10.742 [9.291 – 11.767] peor que sin entrenar | 77.888 [69.051 – 94.393] peor que sin entrenar | 479.784 [440.022 – 521.865] peor que sin entrenar |
| al tope contra el signo, cada uno distinto | 10.742 [9.291 – 11.767] peor que sin entrenar | 86.319 [81.551 – 91.783] peor que sin entrenar | 503.059 [473.879 – 556.159] peor que sin entrenar |
| imita la norma: −delta propio, cada uno distinto | 3.135 [3.126 – 3.140] | 3.312 [3.278 – 3.353] | 14.137 [12.072 – 15.542] peor que sin entrenar |

**Agregador: mediana.** Sin veneno: 3.101 [3.090 – 3.121].

| Ataque | k=1 | k=3 | k=5 |
|---|---|---|---|
| bruto: −10 × delta, todos iguales | 3.128 [3.123 – 3.136] | 3.322 [3.256 – 3.372] | 28399.799 [27169.390 – 29587.513] peor que sin entrenar |
| al tope contra el signo, todos iguales | 3.130 [3.127 – 3.134] | 3.345 [3.311 – 3.373] | 1673.516 [1556.258 – 1865.764] peor que sin entrenar |
| al tope contra el signo, cada uno distinto | 3.130 [3.127 – 3.134] | 3.259 [3.243 – 3.277] | 1820.158 [1736.667 – 1869.687] peor que sin entrenar |
| imita la norma: −delta propio, cada uno distinto | 3.122 [3.114 – 3.130] | 3.240 [3.220 – 3.264] | 127.574 [121.248 – 132.587] peor que sin entrenar |

**Agregador: un voto por cúmulo de copias + mediana.** Sin veneno: 3.101 [3.090 – 3.121].

| Ataque | k=1 | k=3 | k=5 |
|---|---|---|---|
| bruto: −10 × delta, todos iguales | 3.128 [3.123 – 3.136] | 3.148 [3.137 – 3.159] | 3.215 [3.211 – 3.217] |
| al tope contra el signo, todos iguales | 3.130 [3.127 – 3.134] | 3.146 [3.136 – 3.164] | 3.230 [3.224 – 3.235] |
| al tope contra el signo, cada uno distinto | 3.130 [3.127 – 3.134] | 3.259 [3.243 – 3.277] | 1820.158 [1736.667 – 1869.687] peor que sin entrenar |
| imita la norma: −delta propio, cada uno distinto | 3.122 [3.114 – 3.130] | 3.240 [3.220 – 3.264] | 127.574 [121.248 – 132.587] peor que sin entrenar |

**Comparaciones que faltaban.** Cinco honestos solos, sin los tres envenenadores, a 25 rondas: 3.105 [3.092 – 3.127].

**Horizonte largo (mediana, ataque «al tope, todos iguales»).**

| Rondas | Sin veneno | k=1 | k=3 |
|---|---|---|---|
| 60 | 2.928 [2.919 – 2.943] | 2.971 [2.963 – 2.981] | 3.293 [3.281 – 3.311] |

## D. Atribución de la mejora y copias

Montaje: tras 5 rondas honestas, una ronda con 6 aprendices honestos. Recompensa de cada identidad = cuánto empeora el control si se la quita («dejar uno fuera», con mediana). El actor X es el que más aporta y presenta 1, 2, 3 u 8 copias exactas de su delta (se supone que pasaron la verificación: es el caso de datos propios de F4). Cada cifra es media ± desviación sobre 40 sorteos de 128 ventanas de control. «Cuota» = parte de X en la suma de recompensas positivas.

| Copias de X | Suma de recompensas de X (nats) | Cuota de X | Sorteos en que X cobra más que con 1 copia | Honestos con recompensa media negativa |
|---|---|---|---|---|
| 1 | +0.0121 ± 0.0019 | 43 % ± 7 | — | 2 de 5 |
| 2 | +0.0128 ± 0.0040 | 56 % ± 14 | 23 de 40 | 1 de 5 |
| 3 | +0.0003 ± 0.0057 | 8 % ± 10 | 0 de 40 | 0 de 5 |
| 8 | +0.0000 ± 0.0000 | 0 % ± 0 | 0 de 40 | 0 de 5 |

**Coste (una medición).** «Dejar uno fuera» con 6 identidades son 7 evaluaciones: 1038 ms. Una ronda de entrenamiento de 8 aprendices: 429 ms de reloj en paralelo. Un Shapley con 24 permutaciones serían 144 evaluaciones, unas 20 veces más.

