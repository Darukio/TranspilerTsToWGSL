import type { Expr, KernelIR } from "../ir/ir-contract.js";
import { asignarEjes } from "./Bindings.js";

/** Lado de la baldosa: cada workgroup es de BALDOSA × BALDOSA hilos (16 × 16 = 256, el mínimo que WebGPU garantiza). */
export const BALDOSA = 16;

/** Nombres que el kernel con baldosas declara por su cuenta; si el kernel ya usa alguno, no se optimiza. */
export const NOMBRES_INTERNOS = ["tileA_", "tileB_", "lx_", "ly_", "filaOk_", "colOk_", "K_", "kk_", "t_", "ka_", "kb_", "lid"] as const;

/**
 * Una multiplicación de matrices reconocida en el IR:
 *   acc = init;  for k in [0, K): acc = acc + A[fila·SA + k] · B[k·SB + col];  C[…] = acc
 * con `fila` y `col` los dos iteradores del grid.
 */
export interface PlanBaldosas {
  fila: { iter: string; size: Expr };
  col: { iter: string; size: Expr };
  acumulador: { id: string; init: Expr };
  k: { iter: string; hasta: Expr };
  a: { buffer: string; indice: Expr };
  b: { buffer: string; indice: Expr };
  salida: { buffer: string; indice: Expr };
}

// ---- utilidades sobre el IR ----

function visitar(e: Expr, f: (x: Expr) => void): void {
  f(e);
  switch (e.op) {
    case "read": visitar(e.index, f); break;
    case "bin": visitar(e.l, f); visitar(e.r, f); break;
    case "unary": case "cast": visitar(e.x, f); break;
    case "call": e.args.forEach((a) => visitar(a, f)); break;
    case "select": visitar(e.cond, f); visitar(e.a, f); visitar(e.b, f); break;
  }
}

const refs = (e: Expr): Set<string> => {
  const ids = new Set<string>();
  visitar(e, (x) => { if (x.op === "ref") ids.add(x.id); });
  return ids;
};
const tieneLecturas = (e: Expr): boolean => {
  let hay = false;
  visitar(e, (x) => { if (x.op === "read") hay = true; });
  return hay;
};
const esRef = (e: Expr, id: string): boolean => e.op === "ref" && e.id === id;

/** La expresión solo depende de `permitidos` (y de literales) y no lee ningún buffer. */
const soloDe = (e: Expr, permitidos: Set<string>): boolean => !tieneLecturas(e) && [...refs(e)].every((id) => permitidos.has(id));

/** Reemplaza las referencias a `id` por `nuevo` (devuelve una copia). */
export function sustituir(e: Expr, id: string, nuevo: string): Expr {
  switch (e.op) {
    case "ref": return e.id === id ? { op: "ref", id: nuevo } : e;
    case "lit": return e;
    case "read": return { ...e, index: sustituir(e.index, id, nuevo) };
    case "bin": return { ...e, l: sustituir(e.l, id, nuevo), r: sustituir(e.r, id, nuevo) };
    case "unary": case "cast": return { ...e, x: sustituir(e.x, id, nuevo) };
    case "call": return { ...e, args: e.args.map((a) => sustituir(a, id, nuevo)) };
    case "select": return { ...e, cond: sustituir(e.cond, id, nuevo), a: sustituir(e.a, id, nuevo), b: sustituir(e.b, id, nuevo) };
  }
}

/** ¿`idx` es `unitario + escalado · S` (en cualquier orden) con S solo de parámetros? Devuelve true si calza. */
function esAfin(idx: Expr, unitario: string, escalado: string, params: Set<string>): boolean {
  if (idx.op !== "bin" || idx.operator !== "+") return false;
  for (const [u, p] of [[idx.l, idx.r], [idx.r, idx.l]] as const) {
    if (!esRef(u, unitario) || p.op !== "bin" || p.operator !== "*") continue;
    for (const [it, paso] of [[p.l, p.r], [p.r, p.l]] as const) {
      if (esRef(it, escalado) && soloDe(paso, params)) return true;
    }
  }
  return false;
}

function todosLosNombres(kernel: KernelIR): string[] {
  const nombres = [...kernel.params.map((p) => p.name), ...kernel.buffers.map((b) => b.name), ...kernel.grid.map((g) => g.iter)];
  const stmts = (lista: KernelIR["body"]): void => {
    for (const s of lista) {
      if (s.op === "var") nombres.push(s.id);
      if (s.op === "for") { nombres.push(s.iter); stmts(s.body); }
    }
  };
  stmts(kernel.body);
  return nombres;
}

/**
 * Reconoce una multiplicación de matrices en f32 y devuelve su plan, o `undefined` si el kernel no calza EXACTAMENTE
 * con el patrón. Ante cualquier duda no se optimiza: el kernel ingenuo siempre es correcto.
 */
export function detectarBaldosas(kernel: KernelIR): PlanBaldosas | undefined {
  if (kernel.grid.length !== 2 || kernel.body.length !== 3) return undefined;
  const [decl, bucle, almacen] = kernel.body as [KernelIR["body"][number], KernelIR["body"][number], KernelIR["body"][number]];
  if (decl.op !== "var" || decl.type !== "f32" || bucle.op !== "for" || almacen.op !== "store") return undefined;
  if (bucle.from.op !== "lit" || bucle.from.value !== 0 || bucle.body.length !== 1) return undefined;

  const paso = bucle.body[0]!;
  if (paso.op !== "set" || paso.id !== decl.id) return undefined;
  const v = paso.value;
  if (v.op !== "bin" || v.operator !== "+") return undefined;
  const producto = esRef(v.l, decl.id) ? v.r : esRef(v.r, decl.id) ? v.l : undefined;
  if (!producto || producto.op !== "bin" || producto.operator !== "*") return undefined;

  const params = new Set(kernel.params.map((p) => p.name));
  const acc = decl.id;
  const k = bucle.iter;

  // Cota del bucle y valor inicial: uniformes para todo el workgroup (las barreras lo exigen) y sin leer buffers.
  if (!soloDe(bucle.to, params)) return undefined;
  if (!soloDe(decl.init, new Set([...params, kernel.grid[0]!.iter, kernel.grid[1]!.iter]))) return undefined;
  if (!esRef(almacen.value, acc)) return undefined;

  const accesoLeido = (kernel.buffers.find((b) => b.name === almacen.buffer));
  if (!accesoLeido || accesoLeido.access === "read") return undefined;

  for (const [filaIdx, colIdx] of [[0, 1], [1, 0]] as const) {
    const fila = kernel.grid[filaIdx]!;
    const col = kernel.grid[colIdx]!;
    for (const [ra, rb] of [[producto.l, producto.r], [producto.r, producto.l]] as const) {
      if (ra.op !== "read" || rb.op !== "read") continue;
      if (!esAfin(ra.index, k, fila.iter, params) || !esAfin(rb.index, col.iter, k, params)) continue;
      if (!soloDe(almacen.index, new Set([...params, fila.iter, col.iter]))) continue;

      // Los dos operandos son buffers f32 que el kernel solo lee, distintos del que escribe.
      const bufA = kernel.buffers.find((b) => b.name === ra.buffer);
      const bufB = kernel.buffers.find((b) => b.name === rb.buffer);
      if (!bufA || !bufB || bufA.type !== "f32" || bufB.type !== "f32" || bufA.access !== "read" || bufB.access !== "read") continue;
      if (ra.buffer === almacen.buffer || rb.buffer === almacen.buffer) continue;

      // x del grid debe recorrer las columnas (índice contiguo de B y C): lo decide asignarEjes, tiene que coincidir.
      if (asignarEjes(kernel)[0] !== colIdx) continue;

      // Sin choques con los nombres que declara el kernel optimizado.
      const nombres = new Set(todosLosNombres(kernel));
      if (NOMBRES_INTERNOS.some((n) => nombres.has(n))) return undefined;

      return {
        fila: { iter: fila.iter, size: fila.size },
        col: { iter: col.iter, size: col.size },
        acumulador: { id: acc, init: decl.init },
        k: { iter: k, hasta: bucle.to },
        a: { buffer: ra.buffer, indice: ra.index },
        b: { buffer: rb.buffer, indice: rb.index },
        salida: { buffer: almacen.buffer, indice: almacen.index },
      };
    }
  }
  return undefined;
}
