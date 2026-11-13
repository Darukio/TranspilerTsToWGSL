# Metodología del benchmark CPU vs GPU

La página (`pnpm servidor`) incluye un glosario con la definición de cada columna; esta guía explica el método.

Objetivo: que la comparación sea lo más pareja posible. Cada decisión de abajo existe para no favorecer a ninguno de los dos lados.

## Qué se compara
| Serie | Qué es |
|---|---|
| CPU 1 hilo | La función original (`src/pruebas/*.ts`, compilada por `tsc`), sobre `Float64Array` |
| CPU N hilos | El mismo código repartido entre todos los hilos (`os.availableParallelism()`), con memoria compartida |
| GPU total | Subida + cómputo + lectura, de punta a punta |
| GPU cómputo | Solo el pass de cómputo, medido por la GPU con `timestamp-query` |

**La GPU compite contra la mejor CPU de cada tamaño** (la menor mediana entre 1 hilo y N hilos): un programador razonable no usaría 20 hilos para 1000 elementos.

## Reglas
- **Calentamiento:** la GPU descarta `calentamiento` corridas; la CPU al menos 5 (el JIT de V8 tarda en optimizar).
- **Fases separadas:** primero toda la CPU, después toda la GPU. Así los hilos del driver no interfieren con los tiempos de la CPU.
- **Mediana** de las repeticiones; el mínimo y el máximo se ven al pasar el cursor por la celda.
- **El shader se compila una sola vez** (se informa aparte) y no entra en los tiempos por lanzamiento.
- **El cronómetro de la GPU no contamina el total:** `timestamp-query` necesita una lectura extra, así que el cómputo puro se mide en corridas aparte.
- **Verificación de resultados:** la GPU se compara con la CPU (error relativo máximo, por la diferencia f32/f64), y la CPU multihilo debe dar un resultado **idéntico** a la original; si no, el benchmark se detiene con error en lugar de publicar una línea base inválida.
- **Matrices:** la función original recorre N×N y no se puede cortar por filas, así que la CPU multihilo usa una versión por filas escrita a mano (`src/runtime/cpu-worker.ts`). Es la única parte de la línea base que no es el código original.
- **Modo amortizado:** se sube una vez, se lanzan k kernels seguidos en un solo envío y se lee una vez; la CPU hace k pasadas (k × la mediana de una). El punto de empate es una **estimación lineal** con el menor y el mayor k medidos. Los envíos se limitan a ~0,8 s por el TDR de Windows.

## El caso `matmul` (multiplicación de matrices real)
Es el único caso con mucho cálculo por dato (2·N³ operaciones): el transpilador lo traduce con un **acumulador local** (`let suma = 0`), un bucle interno y `suma += …`, y el resultado se compara con la CPU.
- La **CPU de 1 hilo** es la función original tal cual está escrita (`src/pruebas/matmul.ts`, orden i-j-k). Es lento a propósito: es el código fuente sin optimizar.
- La **CPU de N hilos** reparte filas entre hilos y usa **bloqueo de caché** (bloques de 64 en k y 256 en j), escrita a mano en `src/runtime/cpu-worker.ts`. Suma cada `C[i][j]` en el mismo orden de k que la original, así que el resultado es **idéntico bit a bit**; si no lo fuera, el benchmark se detiene.
- La GPU ejecuta lo que genera el transpilador. Cuando el IR es exactamente una multiplicación de matrices en f32, el emisor genera un kernel con **baldosas de 16×16 en memoria compartida** (`src/backend/Baldosas.ts`); si el patrón no calza al detalle, genera el kernel ingenuo. La página indica cuál se usó en cada caso.
- En tamaños lentos (primera corrida de CPU > 0,4 s) esa serie usa 1 calentamiento y 3 repeticiones; el tooltip de la celda muestra cuántas muestras tuvo.
- Reglas de seguridad: una variable declarada **fuera** del bucle y escrita dentro (una reducción) se rechaza, porque sería una carrera entre hilos.

### Cómo se optimiza y por qué es seguro
- `detectarBaldosas` exige el patrón completo: acumulador `f32`, bucle `k` desde 0 con cota que solo depende de parámetros, `acc = acc + A[fila·SA + k]·B[k·SB + col]`, buffers de entrada solo de lectura, y que `gid.x` recorra las columnas. Cualquier otra forma → kernel ingenuo.
- WGSL exige que todos los hilos del grupo lleguen a `workgroupBarrier()`: por eso el kernel con baldosas no tiene `return` temprano; protege las lecturas y la escritura final con guardas y carga `0.0` fuera de rango (no cambia la suma).
- El lanzamiento lee el tamaño de workgroup **del propio shader** (`leerWorkgroup`), así no puede desincronizarse del emisor.
- Los tests comparan el kernel con baldosas contra la CPU y contra el kernel ingenuo en tamaños que no son múltiplo de 16 y en matrices rectangulares.

## WSL (Vulkan + Dozen) contra Windows nativo (Direct3D 12)
Misma máquina, misma GPU (Radeon 880M), mismo código; medianas en ms. WSL: corridas anteriores en `llvmpipe`/`dzn`; Windows: 11 repeticiones, Dawn sobre Direct3D 12. Medidos en momentos distintos, pero las diferencias son de 10 a 100 veces, muy por encima del ruido.

| Caso (tamaño) | GPU total WSL → Windows | Espera WSL → Win | Lectura WSL → Win | Subida WSL → Win |
|---|---|---|---|---|
| calor 128 | 49.7 → 0.87 | 24.8 → 0.17 | 23.3 → 0.25 | 0.3 → 0.29 |
| calor 1024 | 32.2 → 2.82 | 9.0 → 0.58 | 6.3 → 1.16 | 7.7 → 0.91 |
| calor 2048 | 56.1 → 16.8 | 12.7 → 1.35 | 16.6 → 4.55 | 24.8 → 9.97 |
| nbody 4096 | 70.7 → 1.13 | 35.1 → 0.42 | 40.8 → 0.31 | 0.3 → 0.21 |
| nbody 16384 | 56.5 → 2.86 | 45.4 → 1.90 | 14.9 → 0.53 | 0.3 → 0.23 |
| matmul 512 / 1024 / 2048 | 16.4 / 58.5 / 253 → 2.03 / 7.97 / 181 | | | |

- **Lo que más bajó es la espera (sincronización) y la lectura**; la subida solo mejora de forma notable con datos grandes. Con tamaños chicos la subida casi no cambió.
- **El cómputo también es mucho más rápido**: matmul 1024² pasó de ~28 ms a ~3.6 ms por lanzamiento (de ~77 a ~600 GFLOPS) y n-cuerpos 16384 de ~29 ms a ~2 ms. No era solo mover datos: el camino WSL + Dozen frenaba también el cálculo.
- **En Direct3D 12 el cronómetro de la GPU coincide con el tiempo de pared** (cociente 1,0 a 1,1 en tamaños ≥ 1024): la discrepancia de 4 a 8 veces de arriba era un defecto de Dozen, no de `timestamp-query`.
- Contra la mejor CPU: n-cuerpos 16384 pasa de GPU ×1,9 a **GPU ×36**; el calor sigue perdiendo de punta a punta (CPU ×8,3 en 2048²) porque la subida y la lectura de 16 MB dominan, aunque su cómputo puro sí gana (~×4,5).
- Conclusión práctica: **medir rendimiento en Windows nativo**. Los tests de corrección siguen valiendo en WSL.

## Los casos de simulación
- **`nbody`** (gravedad 2D, O(n²)): cada cuerpo suma la fuerza de todos los demás. Mucho cálculo por dato (`sqrt`, división) y datos de solo lectura: el terreno donde la GPU puede ganar. La CPU de N hilos reparte cuerpos (versión por filas con las mismas operaciones y el mismo orden que la original, idéntica bit a bit). Máximo n = 16384: más allá, un envío único se acercaría al límite de ~2 s de la GPU en Windows.
- **`calor`** (stencil de 5 puntos, doble búfer): un paso de la ecuación del calor sobre el interior de la placa. Es una tarea limitada por memoria (5 lecturas y 1 escritura por celda), así que en una GPU integrada que comparte memoria con la CPU no se espera una gran ventaja. El benchmark mide un paso; el modo amortizado repite el mismo paso sobre los mismos buffers, que cuesta lo mismo que alternar los buffers.
- Para **simular de verdad** (varios pasos) el runtime acepta `lanzarKernel(…, { lanzamientos: pasos, intercambio: ['actual', 'siguiente'] })`: los dos buffers se turnan en cada paso y `bufferFinal(par, pasos)` dice cuál tiene el último estado. Hay tests con pasos pares e impares contra una simulación equivalente en la CPU.

## Simulaciones en vivo (`/simulacion`)

`pnpm servidor` también sirve http://localhost:3000/simulacion. El servidor corre los kernels transpilados en la GPU y manda un cuadro por mensaje (`/api/simulacion/stream`); la página solo dibuja.

- **n cuerpos**: `calcularFuerzas` + `integrarCuerpos` (`src/pruebas/integrar.ts`), 5 pasos por cuadro. Posiciones y velocidades quedan en la GPU; solo `x` e `y` vuelven para dibujar.
- **Calor**: `difundirCalor` con doble búfer, 8 pasos por cuadro y una fuente caliente que se mueve (se estampa en la copia de la CPU y se sube).
- **Matmul**: `multiplicarMatricesReal` calcula `C = A·B` en cada cuadro; `B` desplaza y promedia columnas, así que `C` es la imagen `A` moviéndose y desenfocándose. `B` se arma en la CPU y se sube en cada cuadro.
- Comparten la GPU con el benchmark: una sola corrida a la vez (409 si está ocupada). Los tiempos que muestra la página son informativos; para citar cifras se usa el benchmark.
- Cada simulación tiene un test contra la función original en la CPU (`src/tests/simulacion.test.ts`).

## Qué escrituras se aceptan (regla de inyectividad)
Un kernel solo se emite si se puede **probar** que dos hilos distintos nunca escriben la misma posición (`src/middleend/Inyectividad.ts`). El índice de escritura se descompone en `c₁·i₁ + c₂·i₂ + …` y los pasos deben formar una cadena de base mixta (cada paso cubre el rango del anterior: `i·N + j` con `j < N`, o `(i+1)·N + (j+1)` con `i, j < N-2`). Se siguen las variables locales que nunca se reasignan (`const c = …; a[c]`). Se rechazan, entre otras, `a[i + j]`, `a[i / 2]`, `a[i * 0]`, `a[i * N]` (N podría ser 0) y las dispersiones por datos (`a[b[i]]`). Es una condición suficiente y no necesaria: ante la duda se rechaza.

## Cronómetro de la GPU vs tiempo de pared
`timestamp-query` mide lo que la GPU dice haber tardado. En este equipo (Dozen sobre Direct3D12) **no siempre coincide con el reloj de pared**: en matrices 1024² el cronómetro daba ~0.17 ms por lanzamiento y la pendiente de pared ~0.70 ms (4×), con la misma diferencia usando un pass o un pass por lanzamiento. No se sabe la causa.
- La tabla muestra las dos cifras («GPU cómputo (cronómetro)» y «GPU por lanzamiento (pared)») y marca con ⚠ las filas donde difieren 2 veces o más.
- Para decidir quién gana el cómputo se usa el **mayor** de los dos (valor prudente), siempre que el de pared sea fiable (≥ 0,05 ms; por debajo es ruido). Así la comparación no favorece a la GPU.
- Cuando el tope de seguridad deja solo k=1 (kernels lentos) no hay pendiente de pared; en ese caso se estima como `espera − latencia fija`, donde la latencia fija es la espera mínima del mismo caso (la de su kernel más chico).
- El tope de seguridad por envío (~0,8 s, por el TDR de Windows) se predice con el tiempo de **pared** del k anterior, no con el cronómetro.
- El costo por lanzamiento de pared **varió ~4× entre momentos de la misma sesión** para un mismo kernel 1D (vectores 1M: 0.14 ms por la mañana, ~0.6 ms después), mientras el cronómetro se mantuvo en ~0.14 ms. Se descartó que sea la energía compartida entre CPU e iGPU (correr 20 hilos de CPU justo antes no cambia nada). Posibles causas no verificadas: plan de energía o batería, otras aplicaciones usando la GPU, estado del driver. Conviene medir enchufado y con el resto de las aplicaciones cerradas, y repetir antes de citar un número.

## Optimizaciones de la GPU que sí se aplican (y por qué son legítimas)
- **Asignación de ejes** (`asignarEjes`, en `src/backend/Bindings.ts`): `gid.x` recorre el iterador de paso 1 en memoria, para que hilos vecinos accedan a posiciones vecinas. Mejora la GPU sin cambiar el resultado, y se decide analizando los índices, no por una regla fija. Medido en matrices 1024²: 0.83 ms → 0.16 ms por lanzamiento, con salida idéntica bit a bit.

## Probado y descartado (no cambia nada)
- Tamaño de workgroup (64, 128, 256): diferencia menor al 4%.
- Forma del workgroup 2D, una vez corregido el mapeo de ejes.
- Forma del workgroup 2D con el mapeo viejo en anchos mayores a 64: no se midió (el experimento se pasó del límite de tiempo de la GPU y se canceló).
- Estrategia de subida (`writeBuffer`, `mappedAtCreation`, buffer reutilizado): rangos que se pisan.

## Límites conocidos
- **Algunas filas son ruido de la máquina.** Se observó, por ejemplo, un `lectura` de 40 ms para 32 KB, o un tamaño más chico más lento que uno 16 veces mayor. Si una fila no es monótona, repetirla antes de citarla.
- Una sola máquina, con GPU integrada y el camino WSL + Dozen. El ancho de banda de subida (~1 GB/s) y la latencia fija de sincronización (~3 ms) parecen de ese camino; no se probaron en Windows nativo.
- Un kernel 1D no pasa de 65.535 × 64 ≈ 4,19 M elementos (límite de workgroups por eje).
- El ruido entre sesiones es real (la CPU varió hasta ~1,8× entre dos corridas). Repetir antes de citar un número.
