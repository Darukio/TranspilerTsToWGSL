import type { KernelIR, DataType, AccessMode, Expr, Stmt } from "../ir/ir-contract.js";

export interface BindingParams {
  binding: number;
  campos: { name: string; type: DataType }[];
}

export interface BindingBuffer {
  name: string;
  binding: number;
  type: DataType;
  access: AccessMode;
}

export interface Bindings {
  params?: BindingParams;
  buffers: BindingBuffer[];
}

/**
 * Única fuente de los números de @binding: la usan el emisor WGSL (para escribirlos en el shader)
 * y el runtime (para crear los buffers en el mismo orden). `params` va primero si existe.
 */
export function calcularBindings(kernel: KernelIR): Bindings {
  let siguiente = 0;
  const bindings: Bindings = { buffers: [] };
  if (kernel.params.length > 0) {
    bindings.params = { binding: siguiente++, campos: kernel.params.map((p) => ({ name: p.name, type: p.type })) };
  }
  for (const b of kernel.buffers) {
    bindings.buffers.push({ name: b.name, binding: siguiente++, type: b.type, access: b.access });
  }
  return bindings;
}

// Hilos por workgroup según dimensiones del grid. El emisor lo escribe en el shader y el runtime
// lo usa para calcular cuántos workgroups lanzar: tienen que coincidir.
const TAMANO_WORKGROUP: Record<number, [number, number, number]> = { 1: [64, 1, 1], 2: [8, 8, 1], 3: [4, 4, 4] };

export function tamanoWorkgroup(dimensiones: number): [number, number, number] {
  return TAMANO_WORKGROUP[dimensiones] ?? [64, 1, 1];
}

// ---- Asignación de ejes del grid a gid.x / gid.y / gid.z ----
// Los hilos de un workgroup avanzan primero por x. Si x recorre el índice contiguo en memoria, hilos vecinos
// leen y escriben posiciones vecinas (acceso unido); si no, cada hilo salta N posiciones y la GPU pierde ancho de
// banda. Esta regla vive aquí para que el emisor (qué eje usa cada iterador) y el runtime (cuántos workgroups por
// eje) usen siempre la misma.

type Paso = "unitario" | "grande" | "ausente";

/** ¿Con qué paso recorre `iter` esta expresión de índice? unitario = suma directa (…+ j); grande = multiplicado u otra forma. */
function paso(e: Expr, iter: string, locales: Map<string, Expr>, enCurso: Set<string> = new Set()): Paso {
  switch (e.op) {
    case "lit":
    case "read": // el índice de una lectura anidada se evalúa por separado
      return "ausente";
    case "ref": {
      if (e.id === iter) return "unitario";
      // una variable local que nunca se reasigna vale lo mismo que su definición: const c = i*N + j; a[c]
      const def = locales.get(e.id);
      return def && !enCurso.has(e.id) ? paso(def, iter, locales, new Set([...enCurso, e.id])) : "ausente";
    }
    case "cast":
    case "unary":
      return paso(e.x, iter, locales, enCurso);
    case "bin": {
      const l = paso(e.l, iter, locales, enCurso);
      const r = paso(e.r, iter, locales, enCurso);
      if (e.operator === "+" || e.operator === "-") {
        if (l === "ausente") return r;
        if (r === "ausente") return l;
        return "grande"; // aparece dos veces: el coeficiente no es 1
      }
      if (l === "ausente" && r === "ausente") return "ausente";
      if (e.operator === "*") {
        const [con, otro] = l === "ausente" ? [r, e.l] : [l, e.r];
        return con === "unitario" && otro.op === "lit" && otro.value === 1 ? "unitario" : "grande";
      }
      return "grande";
    }
    case "call":
      return e.args.some((a) => paso(a, iter, locales, enCurso) !== "ausente") ? "grande" : "ausente";
    case "select":
      return [e.cond, e.a, e.b].some((x) => paso(x, iter, locales, enCurso) !== "ausente") ? "grande" : "ausente";
  }
}

function recogerIndices(stmts: Stmt[], salida: Expr[]): void {
  const enExpr = (e: Expr): void => {
    switch (e.op) {
      case "read": salida.push(e.index); enExpr(e.index); break;
      case "bin": enExpr(e.l); enExpr(e.r); break;
      case "unary": case "cast": enExpr(e.x); break;
      case "call": e.args.forEach(enExpr); break;
      case "select": enExpr(e.cond); enExpr(e.a); enExpr(e.b); break;
    }
  };
  for (const s of stmts) {
    switch (s.op) {
      case "store": salida.push(s.index); enExpr(s.index); enExpr(s.value); break;
      case "var": enExpr(s.init); break;
      case "set": enExpr(s.value); break;
      case "for": enExpr(s.from); enExpr(s.to); recogerIndices(s.body, salida); break;
    }
  }
}

/** Variables locales del kernel que se declaran una sola vez y nunca se reasignan, con su definición. */
function localesConstantes(stmts: Stmt[]): Map<string, Expr> {
  const defs = new Map<string, Expr | null>();
  const reasignadas = new Set<string>();
  const recorrer = (lista: Stmt[]): void => {
    for (const s of lista) {
      if (s.op === "var") defs.set(s.id, defs.has(s.id) ? null : s.init);
      else if (s.op === "set") reasignadas.add(s.id);
      else if (s.op === "for") recorrer(s.body);
    }
  };
  recorrer(stmts);
  const r = new Map<string, Expr>();
  for (const [id, def] of defs) if (def && !reasignadas.has(id)) r.set(id, def);
  return r;
}

/**
 * Devuelve `orden`, donde `orden[eje]` es el índice de `kernel.grid` que recorre ese eje (0 = x).
 * x se le da al iterador que más veces aparece con paso 1 en los índices de buffers; si ninguno, se conserva el orden original.
 */
export function asignarEjes(kernel: KernelIR): number[] {
  const orden = kernel.grid.map((_, i) => i);
  if (orden.length < 2) return orden;
  const indices: Expr[] = [];
  recogerIndices(kernel.body, indices);
  const locales = localesConstantes(kernel.body);
  const votos = kernel.grid.map((g) => indices.filter((e) => paso(e, g.iter, locales) === "unitario").length);
  const mejor = votos.indexOf(Math.max(...votos));
  if (votos[mejor] === 0) return orden;
  return [mejor, ...orden.filter((i) => i !== mejor)];
}
