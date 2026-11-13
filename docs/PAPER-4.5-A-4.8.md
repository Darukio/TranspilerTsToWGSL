# Borrador del paper: secciones 4.4 a 4.8

> Redactado a partir del código del repositorio (`src/backend`, `src/runtime`, `src/server`, `src/pruebas`). Cada afirmación técnica tiene su archivo de respaldo. Los puntos marcados con **[COMPLETAR]** necesitan un dato que el repositorio todavía no tiene (ver "Notas para el autor" al final, **leer antes de entregar**).

---

## 4.4 Subconjunto Soportado

El transpilador no traduce TypeScript arbitrario: acepta un subconjunto que se puede ejecutar en paralelo sin carreras de datos y expresar en WGSL. Las restricciones tienen dos orígenes. Las **semánticas** vienen del análisis del middle-end (qué bucles son paralelizables y seguros) y se resumen en 4.4.2 y 4.4.5. Las **sintácticas y de modelo de ejecución** vienen de WGSL y están codificadas en el contrato del IR (`ir-contract.ts`, validado con Zod), descritas en 4.4.3. Todo lo que queda fuera se rechaza con un diagnóstico (4.4.6); nunca se genera código incorrecto en silencio.

### 4.4.1 Unidad de traducción: el bucle paralelizable

La unidad es el bucle `for` contado más externo de una función. Cada iteración se convierte en **un hilo** de la GPU. Si el cuerpo del bucle externo es exactamente otro bucle cuya cota no depende del iterador anterior, ese bucle también pasa a ser una dimensión del grid (hasta 3, que son las dimensiones de `global_invocation_id`). Los bucles que no cumplen quedan como `for` internos de cada hilo.

### 4.4.2 Construcciones aceptadas y su correspondencia en WGSL

| TypeScript | WGSL generado | Condición |
|---|---|---|
| `for (let i = 0; i < n; i++)` externo | `let i = i32(gid.x);` + guarda `if (i >= params.n) { return; }` | inicio en el literal `0`, paso 1 |
| `i <= n` | cota `n + 1` | se normaliza a `<` |
| Bucle interno `for (let k = a; k < b; k++)` | `for (var k = a; k < b; k++) { … }` | paso 1; inicio y cota cualesquiera |
| `a[idx] = e` | `a[idx] = e;` | un solo índice lineal; la escritura debe ser inyectiva (4.4.5) |
| `let s = 0;` dentro del cuerpo | `var s: f32 = 0.0;` | local privada del hilo |
| `s += e`, `-=`, `*=`, `/=` | `s = (s + e);` | solo sobre locales declaradas dentro del kernel; el frontend las normaliza a `s = s op e` |
| `+  -  *  /` | `(l + r)` | siempre entre paréntesis |
| `-x` | no soportado directamente | escribir `0 - x` |
| `x!` (aserción no nula) | se ignora | el frontend lo descarta |
| `a[idx]` en una expresión | `a[idx]` | un solo índice |
| `Math.sqrt  sin  cos  tan  abs  exp  log  floor  ceil  pow  min  max` | `sqrt  sin  …  max` | los nombres coinciden con las funciones nativas de WGSL |
| Parámetro escalar (`n`, `N`, `alfa`) | `params.n` | campo de un `struct` uniforme |

### 4.4.3 Restricciones impuestas por WGSL (contrato del IR)

Antes de emitir, el IR se valida con un esquema Zod (`KernelIRSchema`). El esquema restringe, por construcción:

| Aspecto | Restricción | Motivo en WGSL/WebGPU |
|---|---|---|
| Tipos de dato | solo `i32`, `u32`, `f32` | no hay `f64` en WGSL; no se incluyen `bool`, `f16`, vectores ni matrices |
| Expresiones | `lit`, `ref`, `read`, `bin`, `unary`, `cast`, `call`, `select` | conjunto mínimo con equivalente directo |
| Operadores binarios | `+ - * / < > ==` | no hay `<=`, `>=`, `!=`, `&&`, `||` ni `%` |
| Operadores unarios | `-`, `!` | |
| Sentencias | `var`, `set`, `store`, `for` | no hay `if`, `while`, `return`, `break` ni funciones propias |
| Bucle `for` | `iter` de `desde` a `hasta` exclusivo, paso 1 | se emite como `for (var i = d; i < h; i++)` |
| Grid | entre 1 y 3 dimensiones | `global_invocation_id` es un `vec3` |
| Buffers | `array<T>` unidimensional, índice lineal | |
| Modos de acceso | `read`, `write`, `readwrite` | `write` y `readwrite` se emiten como `read_write`, porque WGSL no tiene storage de solo escritura |
| Parámetros | escalares `i32`, `u32` o `f32` | viajan en un `struct` uniforme |
| Versión | `irVersion` igual a `1` | |

A esto se suman las comprobaciones del emisor y de la tabla de símbolos (4.5.2): identificadores reservados de WGSL, símbolos duplicados o no definidos, y escritura sobre un buffer de solo lectura. Del modelo de ejecución salen además restricciones de forma: el tamaño de workgroup depende de la dimensión (64×1×1, 8×8×1, 4×4×4), un kernel 1D no supera ~4,19 millones de elementos (65.535 workgroups por eje × 64 hilos), el buffer uniforme se redondea a múltiplos de 16 bytes y, en el kernel con baldosas, no puede haber `return` temprano porque `workgroupBarrier()` exige que todos los hilos lleguen a ella.

> Nota de precisión: en el esquema Zod, el nombre de la función de `call` es un `string` cualquiera. La lista blanca de funciones matemáticas la impone el middle-end, no el contrato; el emisor escribe el nombre tal cual.

### 4.4.4 Mapeo de tipos

TypeScript tiene un único tipo numérico (`number`, 64 bits), mientras que WGSL exige tipos concretos y no convierte entre ellos de forma implícita. El middle-end infiere el tipo así (`FaseMiddleEnd`):

| Elemento | Tipo WGSL | Regla |
|---|---|---|
| Datos de los arreglos (`number[]`) | `f32` | siempre; el contenido se convierte de 64 a 32 bits al subirlo |
| Iteradores del grid y de bucles internos | `i32` | |
| Parámetro o local usado como índice o cota de bucle | `i32` | se detecta por uso, no por el tipo declarado |
| Cualquier otro parámetro o local | `f32` | |
| Literal en contexto entero | `i32` | si el valor es entero |
| Literal en otro contexto | `f32` | un entero se emite con `.0` |
| Operación con operandos de distinto tipo | ambos a `f32` | se inserta una conversión explícita `f32(x)` |

Dos consecuencias que el paper debe declarar:

- **Pérdida de precisión.** `f32` tiene unos 7 dígitos decimales frente a ~16 de `f64`. El benchmark cuantifica la diferencia (ver 4.8.4) y no hay aviso de pérdida en la transpilación. Hoy el frontend devuelve siempre `number` para el tipo de una variable (`obtenerTipoVariable` es un *stub*), de modo que la inferencia anterior se basa en el uso.
- **División entera.** Si ambos operandos de una `/` son enteros, WGSL divide como entero. Verificado ejecutando el transpilador: `r[i] = n / m + a[m]!` (con `n` y `m` parámetros usados como cota e índice) genera `f32((params.n / params.m)) + …`, de modo que `7 / 2` da `3` en la GPU y `3,5` en JavaScript. Es una divergencia semántica real del subconjunto; no se detecta ni se avisa.

### 4.4.5 Construcciones rechazadas y motivo

| Construcción | Motivo del rechazo |
|---|---|
| `if`, `while`, `return`, ternario `?:`, comparaciones, `%`, `**` | el IR todavía no modela divergencia de control |
| Escribir una variable declarada **fuera** del bucle (`total += a[i]`) | es una reducción entre hilos: carrera de datos sin sincronización |
| Indexado múltiple `a[i][j]` | el IR usa un solo índice lineal (`a[i * N + j]`) |
| Funciones propias, `Math.PI`, `-x` | no hay equivalente en el IR |
| Bucle externo que no empieza en `0` o no avanza de a `1` | no se puede mapear a `global_invocation_id` |
| Bucle interno con paso distinto de 1 | el IR solo tiene `for` de paso 1 |
| Una local con el nombre de un iterador | lo ocultaría |
| `a[i + j]`, `a[i / 2]`, `a[i * 0]`, `a[i * N]`, `a[b[i]]` | no se puede probar que dos hilos escriban posiciones distintas (inyectividad); se rechaza ante la duda |
| Un arreglo que se escribe con índices distintos, o se lee y escribe en índices distintos | posible dependencia entre iteraciones |
| Funciones de `Math` fuera de la lista | no soportada |

**Regla de inyectividad.** Es la restricción semántica central. Un kernel solo se emite si se puede probar que dos hilos distintos nunca escriben la misma posición. El índice de escritura se descompone en `c₁·i₁ + c₂·i₂ + …` y los pasos deben formar una cadena de base mixta (cada coeficiente cubre el rango del anterior: `i·N + j` con `j < N`). Es una condición **suficiente y no necesaria**: puede rechazar escrituras seguras, pero nunca acepta una con colisión probable. Se prefiere esta asimetría porque un kernel secuencial es siempre correcto y una carrera de datos en la GPU produce resultados incorrectos sin ningún error visible.

### 4.4.6 Diagnósticos

Cada rechazo se registra como un diagnóstico `{ fase, mensaje, ubicacion }` en el contexto del pipeline, con la fase que lo emitió y la posición en el fuente (archivo, línea y columna) cuando existe. El resto de los bucles del archivo se sigue procesando.

| Fase | Qué reporta | Ejemplo real |
|---|---|---|
| `frontend` | errores de tipos de TypeScript | |
| `middleend` | por qué un bucle no es apto | `bucle_0: sentencia no soportada: if (i > 2) { r[i] = 1; }` |
| `middleend` | fallo de la prueba de inyectividad, con el motivo | `bucle_0: la escritura a "r" no se puede probar inyectiva (¿dos hilos escribirían la misma posición?): el índice no depende de «k», así que hilos que solo difieren en «k» escriben la misma posición` |
| `middleend` | reducción entre hilos | `escritura a «total», declarada fuera del bucle: una reducción entre hilos necesitaría sincronización` |
| `backend` | violación detectada al emitir | `Kernel "kernel_bucle_0": Violación de acceso: escritura en buffer de solo lectura "A"` |

El CLI imprime los diagnósticos junto con el conteo `Bucles candidatos: N | aptos para GPU: M`. Un error de sintaxis WGSL que el emisor no detecta (por ejemplo, tipos mezclados) aparece recién cuando el runtime compila el shader (`El shader no compila: …`, ver 4.7); el emisor no incluye un verificador de tipos propio.

---

## 4.5 Generación de WGSL

La generación de código es la última fase del transpilador. Recibe el kernel en una representación intermedia (IR) ya verificada por el middle-end y entrega el texto de un *compute shader* en WGSL. El IR es un contrato tipado (`src/ir/ir-contract.ts`) cuyo esquema se valida en tiempo de ejecución con Zod antes de emitir (`FaseBackEnd`): un IR malformado se rechaza como error de entrada, no como WGSL inválido.

### 4.5.1 Recorrido del IR (Visitor)

El IR se define como dos uniones discriminadas por el campo `op`: expresiones (`lit`, `ref`, `read`, `bin`, `unary`, `cast`, `call`, `select`) y sentencias (`var`, `set`, `store`, `for`). La clase `EmisorWGSL` implementa el patrón Visitor mediante dos métodos recursivos, `emitirExpr` y `emitirStmt`, que despachan con un `switch` exhaustivo sobre `op`. Como TypeScript verifica la exhaustividad, agregar un nodo al IR sin su regla de emisión es un error de compilación, no un fallo silencioso en la GPU.

Cada regla traduce un nodo a su equivalente léxico en WGSL:

| Nodo del IR | Texto WGSL | Decisión de diseño |
|---|---|---|
| `lit` (f32 entero) | `2.0` | WGSL no convierte `2` en `f32` implícitamente |
| `ref` a parámetro | `params.N` | los parámetros viajan en un `struct` uniforme |
| `bin` | `(a + b)` | siempre entre paréntesis: la precedencia queda explícita |
| `read` / `store` | `A[idx]` / `A[idx] = v` | índice lineal sobre `array<T>` |
| `select` | `select(b, a, cond)` | WGSL invierte el orden de los argumentos |
| `for` | `for (var k = d; k < h; k++)` | abre un ámbito de símbolos |

### 4.5.2 Tabla de símbolos y comprobaciones semánticas

Mientras recorre el IR, el emisor mantiene una `TablaSimbolos` con una pila de ámbitos (`src/backend/TablaSimbolos.ts`). Cada `for` abre un ámbito y, al cerrarlo, olvida sus variables, de modo que dos bucles hermanos pueden declarar el mismo nombre. La tabla impide tres clases de error antes de que el texto llegue al compilador de shaders: (i) usar un identificador reservado de WGSL o un nombre interno del emisor (`params`, `gid`), (ii) declarar dos veces un símbolo y (iii) referenciar uno inexistente. Además, el emisor rechaza una escritura sobre un buffer declarado como de solo lectura. Las violaciones se convierten en diagnósticos de fase `backend` y no detienen la transpilación de los demás kernels.

### 4.5.3 Plantilla del compute shader

Todo kernel se emite con la misma estructura de tres partes:

```wgsl
// 1. Encabezado: parámetros y buffers, con su número de @binding
struct Params { n: i32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> salida: array<f32>;
@group(0) @binding(2) var<storage, read> vectorA: array<f32>;
@group(0) @binding(3) var<storage, read> vectorB: array<f32>;

// 2. Punto de entrada: tamaño de workgroup y traducción del grid
@compute @workgroup_size(64, 1, 1)
fn kernel_bucle_0(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  // 3. Guarda para los hilos sobrantes y cuerpo del kernel
  if (i >= params.n) { return; }
  salida[i] = ((vectorA[i] * 3.14) - vectorB[i]);
}
```

*Figura 4.x. Shader generado a partir de `procesarVectores` (`src/pruebas/vectores.ts`).*

La guarda existe porque la GPU lanza workgroups completos: si el grid tiene 1000 elementos y el workgroup 64 hilos, se lanzan 1024 hilos y los 24 sobrantes deben terminar sin escribir.

**Fuente única de los recursos.** El número de cada `@binding` y el tamaño del workgroup (64×1×1 para grids 1D, 8×8×1 para 2D y 4×4×4 para 3D) se calculan en un solo módulo (`Bindings.ts`) que consumen tanto el emisor como el runtime. Esto elimina por construcción la desincronización entre lo que el shader declara y lo que el lado CPU enlaza (relevante para la sección 4.7).

### 4.5.4 Optimizaciones del kernel

Además de la traducción directa, el emisor aplica dos optimizaciones. Ambas buscan lo mismo: que la GPU **espere menos por la memoria**, que suele ser el cuello de botella de estos kernels. Ninguna cambia el resultado; solo cambia el orden y la cantidad de accesos a memoria.

**Optimización 1: asignación de ejes (acceso unido a memoria).** Los hilos de un workgroup avanzan juntos y los vecinos son los que difieren en `x`. Cuando esos vecinos piden datos a memoria, la GPU los junta en un único pedido si las direcciones están consecutivas (*coalescing*). En una matriz guardada como `C[i*N + j]`, los elementos con `j` consecutivo están contiguos, mientras que los de `i` consecutivo están separados por `N` posiciones. Por eso conviene que `gid.x` recorra `j`. La función `asignarEjes` lo decide por análisis y no por una regla fija: cuenta, en todos los índices de acceso a buffers, cuántas veces aparece cada iterador sumado directamente (paso 1) y le da `gid.x` al que más votos tiene. Medido en matrices de 1024×1024 (en WSL): de 0,83 ms a 0,16 ms por lanzamiento, con salida idéntica bit a bit.

**Optimización 2: baldosas en memoria compartida (multiplicación de matrices).** En un producto de matrices cada hilo lee de memoria global una fila de `A` y una columna de `B` enteras, y miles de hilos leen los mismos datos. Con baldosas, el workgroup de 16×16 hilos copia un bloque de `A` y uno de `B` a memoria compartida (`var<workgroup>`, rápida y dentro del chip), sincroniza con `workgroupBarrier()` y todos lo reutilizan 16 veces antes de pasar al bloque siguiente. Se lee de memoria global unas 16 veces menos. Medido (en WSL): matmul 1024² de 83,5 ms a 27,8 ms (×3,0).

| | Asignación de ejes | Baldosas |
|---|---|---|
| Idea | hilos vecinos acceden a posiciones vecinas | cargar una vez y reutilizar 16 veces |
| Se aplica a | cualquier kernel con grid de 2 o 3 dimensiones | solo multiplicación de matrices en `f32` |
| Cómo se decide | votos por iterador sobre los índices | `detectarBaldosas` exige el patrón completo |
| Si no se cumple | se conserva el orden original | se emite el kernel ingenuo, siempre correcto |

La segunda optimización exige reglas de WGSL: todos los hilos del grupo deben llegar a la barrera, por lo que ese kernel no tiene `return` temprano. Protege las lecturas y la escritura final con guardas y carga `0.0` fuera de rango, valor que no altera la suma. El patrón que se reconoce es estricto (acumulador `f32`, bucle `k` desde 0 con cota que solo depende de parámetros, `acc = acc + A[..]·B[..]` y buffers de entrada de solo lectura). Cualquier diferencia cae al kernel ingenuo. Es la única especialización implementada.

### 4.5.5 Alcance y límites

El IR actual admite aritmética `+ - * /`, comparaciones, conversiones, `select`, acumuladores locales, bucles internos y las funciones `sqrt sin cos tan abs exp log floor ceil pow min max`. No hay condicionales, reducciones entre hilos ni generación de números aleatorios; esos casos se rechazan con un diagnóstico que indica la fase y el motivo.

---

## 4.6 Integración con el Flujo de Compilación

### 4.6.1 Transpilación anticipada (AoT)

La transpilación ocurre **antes de la ejecución** del programa: el código TypeScript se analiza y se traduce a WGSL como parte del proceso de construcción, y el shader resultante es un artefacto estático (`wgsl/<kernel>.wgsl`). En tiempo de ejecución no se interpreta ni se genera código; el runtime solo compila el WGSL ya emitido a la GPU. Esto contrasta con enfoques que construyen el shader en ejecución, donde el costo de la traducción se paga en cada arranque.

### 4.6.2 Arquitectura de fases

El transpilador es un *pipeline* de fases con un contexto compartido (`ContextoPipeline`) que acumula bucles candidatos, bucles aptos, kernels IR, shaders y diagnósticos. Se compone con un `PipelineBuilder`:

```
TypeScript → Frontend → NodoBucle → Middle-end → KernelIR → Backend → .wgsl
             (ts.Program)           (aptitud +                (4.5)
                                    inyectividad)
```

*Figura 4.y. Fases del transpilador.* El frontend usa la API del compilador de TypeScript (`ts.createProgram`, a través de `TSCompilerFacade`), por lo que reporta los mismos diagnósticos de tipos que `tsc`. Sin archivos explícitos toma el `tsconfig.json` del proyecto; con archivos explícitos crea el programa con opciones propias. El middle-end decide qué bucles pueden ejecutarse en paralelo sin que dos hilos escriban la misma posición. Cada rechazo queda como diagnóstico etiquetado con la fase que lo emitió.

### 4.6.3 Punto de enganche con `tsc`

La herramienta se publica como binario (`ts-to-wgsl`, `src/cli.ts`) y se encadena al ciclo de construcción con `tsc`:

```
pnpm build         # tsc: compila el proyecto a JavaScript (dist/)
ts-to-wgsl src/pruebas/vectores.ts   # emite wgsl/kernel_bucle_0.wgsl
```

El mismo `tsc` produce la línea base de CPU (el código original compilado a JavaScript, que el benchmark ejecuta tal cual) y el transpilador produce el kernel de GPU a partir del **mismo fuente**. Así, ambas versiones salen de un único archivo `.ts` y no pueden divergir.

> **[Ver nota 1 al final]** Esta integración es un paso de construcción que reutiliza la API del compilador. No es un *plugin* ni un *transformer* que se ejecute dentro de `tsc`. Si el paper afirma lo segundo, hay que implementarlo o reformular el texto.

### 4.6.4 Sobrecarga de la integración

La transpilación de cada caso de prueba (frontend + middle-end + backend, medido de extremo a extremo con un `ts.Program` nuevo por invocación) tarda del orden de **1,1 s** (mediana de 10 corridas por caso: 1,04–1,10 s; ver 4.8.5). Es un costo de construcción, que se paga una vez por compilación y no por ejecución.

---

## 4.7 Puente de Memoria CPU–GPU

### 4.7.1 Elección de la interfaz

En Node.js no existe `navigator.gpu`. El puente usa **Dawn**, la implementación de WebGPU de Chromium, mediante el paquete `webgpu` (bindings nativos): esa librería nativa cumple el rol que el objetivo 4 describe como interfaz de funciones foráneas. En el navegador, el mismo código usa `navigator.gpu` sin cambios (`src/runtime/dispositivo.ts`). El adaptador se pide con `requestAdapter()` y se habilita `timestamp-query` solo si el hardware lo ofrece. El runtime detecta además si el adaptador es un emulador en CPU (`llvmpipe`, `SwiftShader`) y lo marca, porque medir rendimiento contra un emulador no tiene sentido.

### 4.7.2 Ruta de conexión en Windows (Direct3D 12)

WebGPU es una API de alto nivel: no habla directamente con la GPU, sino con la API gráfica nativa del sistema operativo. El proyecto no elige esa API; llama a `requestAdapter()` y es Dawn quien selecciona el *backend* según la plataforma. En las mediciones en Windows, ese backend fue **Direct3D 12** (`docs/BENCHMARK.md`).

```
TypeScript (Node.js)
   │  navigator.gpu no existe en Node → import('webgpu')
   ▼
paquete `webgpu` (bindings nativos de Dawn)
   ▼
Dawn  ── compila WGSL ──▶  shader nativo del backend
   ▼
Direct3D 12
   ▼
controlador de la GPU (AMD Radeon 880M, integrada)
```

*Figura 4.z. Ruta de ejecución en Windows nativo.*

- **Sin configuración adicional.** En Windows basta con Node.js y el controlador de la GPU al día: Direct3D 12 ya forma parte del sistema. No hay variables de entorno ni parches (**[no se verificó la instalación en una máquina limpia]**).
- **Verificación del adaptador.** `pnpm gpu-info` imprime qué GPU detectó Dawn, si es de hardware o emulada, y si soporta `timestamp-query` (`describirDispositivo`). Si el adaptador es un emulador, el benchmark lo marca como no válido para comparar rendimiento.
- **Memoria.** En la GPU integrada, CPU y GPU comparten la misma RAM física. Aun así, la API exige copias explícitas (`writeBuffer`, `copyBufferToBuffer`, `mapAsync`), de modo que el costo de subida y lectura sigue existiendo y se mide por separado (4.8).
- **Límite de tiempo del sistema.** Windows reinicia el controlador (TDR) si un envío a la GPU dura más de ~2 s. Por eso el benchmark limita cada envío a ~0,8 s (4.8.4).

**Contraste con WSL.** Dentro de WSL, Dawn solo habla Vulkan y WSL no expone la GPU directamente. Hace falta una capa intermedia, Dozen (Vulkan sobre Direct3D 12), compilada aparte con un parche local (`docs/GPU-EN-WSL.md`). Esa traducción extra agregó entre 10 y 100 veces de latencia en sincronización y lectura, y volvió más lento el propio cómputo (`docs/BENCHMARK.md`). Por eso el rendimiento se mide en Windows nativo, y WSL se reserva para verificar corrección.

| | Windows nativo | WSL |
|---|---|---|
| Ruta | Node → Dawn → Direct3D 12 → GPU | Node → Dawn → Vulkan → Dozen → Direct3D 12 → GPU |
| Configuración | ninguna | Mesa/Dozen compilado a mano + `source scripts/wsl-gpu.sh` |
| Uso en el proyecto | mediciones de rendimiento | pruebas de corrección |

> Precisión: la traducción interna de Dawn de WGSL al lenguaje del backend no está en el código del proyecto, sino en la librería, y no se inspeccionó. Si el paper la describe en detalle, hay que citar la documentación de Dawn.

### 4.7.3 Ciclo de vida de un lanzamiento

Ejecutar un kernel en la GPU siempre sigue tres pasos: **subir** los datos, **calcular** y **leer** el resultado. Cada paso se mide por separado en el benchmark (4.8), por lo que conviene entender qué ocurre en cada uno. La unidad de memoria de la GPU es el *buffer*: un bloque de tamaño fijo que se reserva con `createBuffer`. El shader solo puede leer y escribir en buffers, no en variables de JavaScript. Además, JavaScript no puede acceder directamente a ellos; toda transferencia se hace con una operación explícita de la API, incluso en una GPU integrada que comparte la RAM con la CPU.

```
CPU                       GPU                       Buffer intermedio (MAP_READ)
 │ 1. writeBuffer(datos) ──▶│                                │
 │    (Float64 → f32)       │                                │
 │ 2. dispatch() ──────────▶│ pass de cómputo                │
 │ 3. ordena la copia ─────▶│ copyBufferToBuffer() ─────────▶│
 │ 4. mapAsync(MAP_READ) ───────────────────────────────────▶│
 │◀──────────────────────────── datos mapeados ──────────────│
```

*Figura 4.w. Los tres pasos de un lanzamiento. La CPU emite las órdenes y la GPU las ejecuta.*

**Paso 1. Subida (`buffers.ts`).** Hay que preparar dos tipos de datos.

- *Arreglos de entrada.* Los arreglos de JavaScript guardan `number` de 64 bits. Se convierten al tipo que declara el kernel (`Float32Array` para `f32`, `Int32Array` para `i32`, `Uint32Array` para `u32`) y se copian a la GPU con `queue.writeBuffer`.
- *Parámetros escalares* (por ejemplo `N` o `alfa`). Son valores sueltos que el kernel necesita y viajan juntos en un buffer pequeño de tipo *uniform*. Para armarlo se escribe cada número como bytes crudos con `DataView`, en el orden de bytes *little-endian* que usan las CPU y GPU comunes. Cada parámetro ocupa 4 bytes y WebGPU exige que el tamaño total del buffer uniform sea múltiplo de 16 bytes, por lo que se completa con ceros. Por ejemplo, dos parámetros ocupan 8 bytes y se reservan 16.
- *Buffers de salida.* No se suben datos. Basta indicar su largo: la GPU los crea llenos de ceros y el kernel los completa.

Antes de reservar memoria en la GPU, se valida la entrada completa (buffers faltantes o vacíos, parámetros ausentes). Así un error de uso se detecta en la CPU, sin dejar recursos a medio crear.

**Paso 2. Cómputo (`dispatch.ts`).** Esta etapa tiene dos partes.

- *Preparación, una sola vez.* `prepararKernel` compila el shader y crea el *pipeline*. Su costo se informa aparte y no se repite por lanzamiento.
- *Lanzamiento, tantas veces como se quiera.* La GPU ejecuta los hilos en grupos (*workgroups*) de tamaño fijo, así que hay que indicar cuántos grupos lanzar por cada eje: `⌈tamaño / hilos por grupo⌉`. Con 1000 elementos y grupos de 64 hilos hacen falta 16 grupos (1024 hilos; los 24 sobrantes no escriben, gracias a la guarda de 4.5.3). El tamaño del grupo se **lee del propio shader** (`leerWorkgroup`) y no se recalcula, para que el lanzamiento no pueda diferir de lo que escribió el emisor. Antes de lanzar se comprueba que el número de grupos no supere el límite de la GPU por eje.

**Paso 3. Lectura (`lectura.ts`).** Los buffers donde el kernel escribe (*storage*) no se pueden abrir desde JavaScript. El resultado se copia primero a un *buffer intermedio* marcado `MAP_READ` (`copyBufferToBuffer`), que sí admite lectura. Luego `mapAsync` lo deja accesible a la CPU y se copia su contenido (`slice`) antes de destruir el intermedio. Este paso espera a que la GPU termine, por lo que su costo incluye parte de la sincronización.

| Paso | Operación de la API | Qué se mide en 4.8 |
|---|---|---|
| 1. Subida | `writeBuffer` | `subida` |
| 2. Cómputo | `dispatchWorkgroups`, `onSubmittedWorkDone` | `espera` y `GPU cómputo` |
| 3. Lectura | `copyBufferToBuffer`, `mapAsync` | `lectura` |

### 4.7.4 Seguridad de memoria

Como el objetivo pide ausencia de fugas y de fallos de segmentación, el puente aplica las siguientes medidas:

- **Validación previa a la GPU.** `crearBuffers` valida la entrada completa (buffers faltantes, vacíos, de solo lectura sin datos, parámetros ausentes) antes de reservar nada en la GPU.
- **Liberación determinista.** Todo buffer creado se registra y `BuffersGPU.destruir()` los libera todos; el benchmark lo invoca en un bloque `finally`, igual que la destrucción del buffer intermedio de lectura y del `QuerySet` de timestamps. Una excepción a mitad de una medición no deja recursos reservados.
- **Errores de WebGPU como excepciones.** `conValidacion` envuelve las operaciones en un *error scope* de validación y convierte el error asíncrono de la GPU en una excepción de JavaScript. El resultado de compilar el shader también se revisa (`getCompilationInfo`) antes de crear el pipeline.
- **Layout explícito.** El `bindGroupLayout` se declara a mano porque con `layout: 'auto'` WebGPU descarta los bindings que el shader no usa.
- **Referencias retenidas.** Se detectó un fallo de segmentación intermitente (`futex…`) causado por el recolector de basura: el objeto `gpu` y el `adapter`, al ser variables locales, se liberaban mientras el dispositivo seguía en uso. Ahora se retienen en un conjunto hasta `destruir()` (`dispositivo.ts`). Este caso es la evidencia concreta de que el límite entre el heap de JavaScript y la memoria nativa necesita gestión explícita.
- **Aislamiento en el servidor del benchmark.** Solo atiende a `127.0.0.1`/`localhost` (evita DNS *rebinding*), exige una cabecera propia para lanzar trabajo y permite una sola corrida a la vez, porque compartirían la GPU y falsearían los tiempos.

### 4.7.5 Modo de datos residentes

Para evitar pagar subida y lectura en cada paso, `lanzarKernel` acepta `lanzamientos: k` (k pasadas seguidas en un solo envío) y `intercambio: [a, b]`, que alterna dos buffers como doble búfer: en los pasos pares el kernel lee `a` y escribe `b`, en los impares al revés. `bufferFinal(par, pasos)` indica dónde quedó el último estado. Esto modela una simulación real (por ejemplo, difusión de calor con varios pasos) donde los datos permanecen en la GPU.

---

## 4.8 Diseño Experimental

### 4.8.1 Casos de prueba

Los seis casos están en `src/pruebas/` y son los mismos fuentes que el transpilador consume y que `tsc` compila como línea base de CPU.

| Caso | Cálculo | Carga por dato | Tamaños de entrada |
|---|---|---|---|
| `vectores` | `salida[i] = A[i]·3,14 − B[i]` | baja (acotado por memoria) | 10³, 10⁴, 10⁵, 10⁶, 4·10⁶ |
| `matrices` | `C[i,j] = A[i,j] + B[i,j]` (suma, 2D) | baja | N = 32, 128, 256, 512, 1024 |
| `monte-carlo` | `√(p·2,5) + sin(p)` por muestra | media (funciones matemáticas) | 10³, 10⁴, 10⁵, 10⁶, 4·10⁶ |
| `matmul` | `C = A·B` con acumulador local | alta (2N³ operaciones) | N = 64, 128, 256, 512, 1024, 2048 |
| `nbody` | gravedad 2D entre todos los pares, O(n²) | alta (`sqrt`, división) | n = 256, 1024, 4096, 16384 |
| `calor` | un paso de la ecuación del calor (stencil de 5 puntos) | baja (5 lecturas, 1 escritura) | N = 128, 256, 512, 1024, 2048 |

Los casos cubren tres regímenes: acotados por memoria (`vectores`, `matrices`, `calor`), con muchos cálculos por dato (`matmul`, `nbody`) e intermedios (`monte-carlo`). El caso `monte-carlo` evalúa la parte aritmética de una simulación de precios (valor determinista por muestra); **[ver nota 2]**.

### 4.8.2 Líneas base

| Serie | Descripción |
|---|---|
| CPU 1 hilo | La función original, compilada por `tsc`, sobre `Float64Array` |
| CPU N hilos | El mismo código repartido entre `os.availableParallelism()` hilos (`worker_threads`) con memoria compartida (`SharedArrayBuffer`) |
| GPU total | Subida + cómputo + lectura, de punta a punta |
| GPU cómputo | Solo el *pass* de cómputo, medido con `timestamp-query` |

La GPU se compara contra la **mejor CPU de cada tamaño** (la menor mediana entre 1 hilo y N hilos): un programador razonable no usaría 20 hilos para 1000 elementos, y comparar contra la peor CPU inflaría el resultado. En `matrices` la función original no puede cortarse por filas, por lo que la versión multihilo es una reescritura por filas (`cpu-worker.ts`), la única parte de la línea base que no es el código original; en `matmul` esa versión usa además bloqueo de caché. Para que la línea base sea válida, el benchmark exige que la CPU multihilo produzca un resultado **idéntico bit a bit** al de la original; si no, aborta con error. Esta verificación ya detectó un error real de reparto de filas en `calor`.

### 4.8.3 Métricas

| Métrica | Definición | Cómo se mide |
|---|---|---|
| Tiempo de ejecución | Mediana del tiempo de cada serie | `performance.now()`; `GPU cómputo` con `timestamp-query` |
| Latencia de transferencia | Subida (CPU→GPU), espera (sincronización) y lectura (GPU→CPU) | Cronometradas por separado en cada corrida |
| Sobrecarga de transpilación | Tiempo de TypeScript a WGSL; tiempo de compilación del shader | Ver 4.8.5; `msCompilacion` (`prepararKernel`) |
| Error numérico | Mayor error relativo GPU(f32) vs CPU(f64) | Ver 4.8.4 |
| Punto de empate | Menor k a partir del cual la GPU gana con datos residentes | Ajuste lineal `total(k) ≈ fijo + k·porLanzamiento` con el menor y mayor k medidos |

### 4.8.4 Repeticiones, calentamiento y tolerancia numérica

- **Repeticiones.** 15 por defecto (5 en modo rápido), con **mediana** como estadístico; mínimo, máximo y media quedan registrados.
- **Calentamiento.** 3 corridas descartadas en la GPU (1 en modo rápido) y al menos 5 en la CPU, porque el compilador JIT de V8 tarda en optimizar y dejaría a la CPU en desventaja. Si la primera corrida de CPU supera 0,4 s, esa serie usa 1 calentamiento y 3 repeticiones para que el benchmark termine.
- **Aislamiento.** Primero se mide toda la CPU y luego toda la GPU, para que los hilos del controlador no interfieran con los tiempos de la CPU. El cronómetro de la GPU se mide en corridas aparte, porque leerlo cuesta una lectura extra que contaminaría el total.
- **Shader compilado una vez.** Su costo se informa aparte y no entra en los tiempos por lanzamiento.
- **Tolerancia.** La GPU calcula en `f32` y la CPU en `f64`, por lo que la diferencia es inevitable. Se mide `|gpu − cpu| / max(1, |cpu|)`, y los tests de integración fijan un umbral por caso:

| Caso | Tolerancia relativa máxima |
|---|---|
| `matrices` | 10⁻⁶ |
| `vectores`, `matmul`, `calor` | 10⁻⁵ |
| `nbody`, `monte-carlo` | 10⁻³ |

Los umbrales más holgados corresponden a casos con sumas largas de términos de signo mixto en `f32` (`nbody`) y a funciones trascendentes, cuya precisión en GPU no es exactamente la de `Math.*` (`monte-carlo`).

- **Tope por envío.** Un envío a la GPU se limita a ~0,8 s (se predice con el tiempo de pared del k anterior): Windows reinicia el controlador (TDR) si un envío pasa de ~2 s.

### 4.8.5 Sobrecarga de transpilación

Se midió el transpilador completo (`new TranspiladorTsaWGSL().transpilar([archivo])`, un `ts.Program` nuevo por invocación) con 1 corrida de arranque y 10 corridas de medición por caso.

| Caso | Mediana (ms) | Mínimo (ms) | Máximo (ms) |
|---|---|---|---|
| `vectores` | 1112 | 995 | 1262 |
| `matrices` | 1102 | 992 | 1132 |
| `monte-carlo` | 1096 | 993 | 1274 |
| `matmul` | 1096 | 980 | 1343 |
| `nbody` | 1099 | 988 | 1314 |
| `calor` | 1038 | 962 | 1175 |

El tiempo es prácticamente igual en los seis casos, lo que indica que el costo no depende del tamaño del kernel. Se midió en WSL2 (20 hilos); **[COMPLETAR: repetir en Windows nativo y desglosar cuánto corresponde a la creación del `ts.Program` frente al middle-end y el backend; esa descomposición no se midió]**.

### 4.8.6 Entorno

Una sola máquina: AMD Ryzen AI 9 365 con GPU integrada Radeon 880M (memoria compartida con la CPU), 20 hilos de CPU, Node.js con Dawn como implementación de WebGPU. **El rendimiento se mide en Windows nativo (Direct3D 12).** El camino WSL (Vulkan sobre Dozen) agregó entre 10 y 100 veces de latencia en sincronización y lectura, y su cómputo era más lento (`docs/BENCHMARK.md`); WSL se usa solo para verificar corrección. La suite de pruebas automatizadas verifica la corrección en WSL sobre `llvmpipe` (emulador) si no hay GPU real.

### 4.8.7 Resultados preliminares (medianas en ms; fuente: `docs/BENCHMARK.md`)

La tabla siguiente reproduce lo ya medido para justificar el diseño y debe **reemplazarse** por la corrida final completa de los seis casos en Windows nativo, con la máquina enchufada y las demás aplicaciones cerradas.

| Caso (tamaño) | GPU total, WSL → Windows | Espera, WSL → Windows | Lectura, WSL → Windows |
|---|---|---|---|
| calor (128) | 49,7 → 0,87 | 24,8 → 0,17 | 23,3 → 0,25 |
| calor (2048) | 56,1 → 16,8 | 12,7 → 1,35 | 16,6 → 4,55 |
| nbody (4096) | 70,7 → 1,13 | 35,1 → 0,42 | 40,8 → 0,31 |
| nbody (16384) | 56,5 → 2,86 | 45,4 → 1,90 | 14,9 → 0,53 |
| matmul (512 / 1024 / 2048) | 16,4 / 58,5 / 253 → 2,03 / 7,97 / 181 | — | — |

Observaciones ya respaldadas: en `nbody` 16384 la GPU supera a la mejor CPU por un factor de ~36 en Windows; en `calor` la CPU gana de punta a punta (×8,3 en 2048²) porque subir y leer 16 MB domina sobre un cálculo acotado por memoria, aunque el cómputo puro de la GPU sí gana (~×4,5). El kernel con baldosas redujo `matmul` 1024² de 83,5 ms a 27,8 ms (×3,0) en el entorno WSL.

### 4.8.8 Amenazas a la validez

- **Una sola máquina y una GPU integrada**: los resultados no se generalizan a GPU discretas, donde el costo de transferencia por el bus PCIe cambia las conclusiones.
- **Ruido entre sesiones**: se observaron variaciones de hasta ~4× en el costo por lanzamiento de un mismo kernel entre momentos de la misma sesión, y la CPU varió hasta ~1,8× entre corridas. Causas no verificadas (plan de energía, otras aplicaciones, estado del controlador); por eso se reporta la mediana y se recomienda repetir antes de citar una cifra.
- **Cronómetro de la GPU**: en el camino WSL discrepó del tiempo de pared hasta 4–8×; en Direct3D 12 coinciden (cociente 1,0–1,1 en tamaños ≥ 1024). Para decidir quién gana el cómputo se usa el mayor de ambos valores (criterio prudente).
- **Precisión**: `f32` en GPU frente a `f64` en CPU; el error se cuantifica y acota (4.8.4).
- **Kernels admitidos**: sin condicionales, reducciones ni aleatoriedad, el conjunto de programas cubierto es acotado.

---

## Notas para el autor (borrar antes de entregar)

Estas son las diferencias entre lo que piden las secciones/objetivos y lo que el repositorio realmente tiene. Conviene resolverlas o reformular el texto, porque un revisor las va a encontrar.

1. **No hay integración *dentro* de `tsc`.** Lo implementado es un CLI que usa la API del compilador de TypeScript y se encadena después de `tsc` en la construcción. Eso sí es AoT, pero no es un plugin/transformer. Opciones: (a) mantener el texto como está (es honesto); (b) implementar un transformer (p. ej. con `ts-patch`) si el paper necesita afirmar "se engancha en tsc". Además, `src/api/` y `src/index.ts` existen pero están **vacíos** (sin seguimiento en git).
2. **`monte-carlo` no es un Monte Carlo.** `src/pruebas/monte-carlo.ts` calcula `sqrt(p·2.5) + sin(p)` por elemento, sin números aleatorios ni agregación. El IR no soporta aleatoriedad ni reducciones (límites declarados en `docs/MIDDLEEND-Y-BACKEND.md`). Si el paper lo llama "Monte Carlo" sin matiz, es engañoso: o se renombra (p. ej. "evaluación de funciones matemáticas") o se agrega un generador pseudoaleatorio (p. ej. PCG por hilo) y una reducción.
3. **TypeGPU queda fuera de las líneas base** (decisión del autor): el diseño compara solo CPU (1 hilo y N hilos) contra GPU. Conviene justificarlo en el paper, porque el enunciado original de la sección lo mencionaba.
4. **La sobrecarga de transpilación no estaba medida por el benchmark.** El benchmark solo registra la compilación del shader en WebGPU (`msCompilacion`). La tabla de 4.8.5 la medí para este documento con un script aparte (no está en el repositorio): conviene incorporar esa medición a `benchmark.ts` para que sea reproducible. Está medida en WSL, y el desglose (`ts.Program` vs. fases propias) no se hizo.
5. **Resultados:** las únicas cifras con respaldo son las de `docs/BENCHMARK.md` y corresponden a corridas parciales en dos entornos. Falta una corrida final completa (6 casos × todos los tamaños) en Windows nativo, y la tabla 4.8.7 no debe presentarse como resultado final.
6. **Puente de memoria / objetivo 4 ("sin fugas ni errores de segmentación").** Hay liberación determinista y se corrigió un segfault real, pero no hay una prueba específica de fugas (p. ej. repetir miles de ciclos y comprobar que la memoria se estabiliza). Sin ella, el texto no debería afirmar "sin fugas", solo describir las medidas tomadas. Tampoco se usa FFI propio: se usa la librería nativa de Dawn.
7. **"Visitor":** el emisor aplica el patrón con recorrido recursivo y `switch` sobre uniones discriminadas, no con una clase `Visitor` con métodos `visitX` al estilo GoF. El texto lo describe con esa precisión.
8. **Cifras de matrices:** `docs/BENCHMARK.md` cita 0,83 → 0,16 ms con la asignación de ejes, y `docs/MIDDLEEND-Y-BACKEND.md` cita 83,5 → 27,8 ms con baldosas para matmul 1024²; ambas son de WSL, anteriores a las de Windows. No mezclarlas con la tabla de Windows.
