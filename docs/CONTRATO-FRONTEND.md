# Contrato del Frontend

Guía para quien trabaja en `src/frontend/`. Explica qué debe entregar el frontend y qué espera el resto del pipeline.

## Dónde estás en el pipeline

```
Frontend  →  NodoBucle  →  Middle-end  →  KernelIR (Zod)  →  Backend  →  .wgsl
(vos)        Modelo.ts                    ir-contract.ts
```

El frontend **no** produce el `KernelIR`. Eso lo arma el middle-end (`src/middleend/FaseMiddleEnd.ts`). Tu salida son los tipos de `src/core/Modelo.ts`.

## Qué tenés que implementar

La interface `IGestorAST` (`src/core/IGestorAST.ts`):

| Método | Debe devolver |
|---|---|
| `cargar(archivos)` | Nada. Prepara el `ts.Program`. Si `archivos` está vacío, usa el `tsconfig.json` local. |
| `obtenerBuclesCandidatos()` | `NodoBucle[]`: solo los bucles **más externos**. Los anidados van dentro de `cuerpo`, no duplicados como candidatos. |
| `obtenerDiagnosticos()` | Errores de TypeScript como `Diagnostico[]`. |
| `obtenerTipoVariable(bucleId, nombreVar)` | Tipo de la variable, o `undefined`. **Hoy es un stub** (ver "Pendiente"). |

## Forma de los datos (`Modelo.ts`)

Un bucle candidato es un `for` contado: `for (let i = a; i < b; i++)`, también `<=` y `i += expr`.

```ts
NodoBucle { id, ubicacion, variable, inicio, limite, inclusivo, paso, cuerpo }
Expr      = num | var | bin(+ - * /) | acceso | llamada | desconocida
Sentencia = asignacion | bucle | desconocida
```

Reglas que el middle-end da por hechas:

- `acceso` lleva **todos** los índices en `indices`: `a[i][j]` → `[i, j]`.
- `llamada` usa el nombre completo: `Math.sqrt`, no `sqrt`.
- Lo que no se entiende se devuelve como `desconocida` con su `texto`. **No lo omitas ni lances un error**: el middle-end lo usa para explicar por qué descarta el bucle.
- `inclusivo` es `true` para `<=`.

## Qué considera "apto" el middle-end

Para que un bucle llegue a la GPU, el middle-end exige:

- El bucle externo empieza en `0` con paso `1`.
- El cuerpo solo tiene asignaciones a elementos de arreglo (`a[idx] = ...`) y bucles internos con paso `1`.
- Los accesos usan **un solo índice** (índice lineal).
- Cada escritura depende de todas las variables del grid. Un arreglo que se lee y se escribe con índices distintos se rechaza.
- Solo se admiten `+ - * /` y estas funciones: `Math.sqrt sin cos tan abs exp log floor ceil pow min max`.

Todo lo demás se descarta con un diagnóstico de fase `middleend`.

## Pendiente conocido

- `obtenerTipoVariable` devuelve siempre `'number'`.
- TypeScript no distingue `i32`, `u32` o `f32`: todo es `number` (64 bits). Hoy el middle-end asume `f32` para datos y `i32` para variables usadas como índice o límite. No hay aviso de pérdida de precisión.
- Sentencias como `if`, `let` o `+=` caen como `desconocida`.

## Cómo probar

```bash
pnpm test                           # compila y corre los tests
pnpm build && node dist/cli.js src/pruebas/vectores.ts
```

Los casos de entrada están en `src/pruebas/`.
