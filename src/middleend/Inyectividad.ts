import type { Expr } from "../core/Modelo.js";

/**
 * Prueba de que una escritura `a[indice]` no tiene colisiones entre hilos: que la función
 * (i₁, …, iₘ) → indice sea inyectiva sobre el dominio 0 ≤ iₖ < Sₖ.
 *
 * Método (base mixta): se descompone el índice en  c₁·i₁ + c₂·i₂ + … + (término sin iteradores)  y se busca un orden
 * de los iteradores tal que
 *   - el coeficiente del primero es una constante positiva, y
 *   - cada coeficiente siguiente es, al menos, el anterior multiplicado por el tamaño del anterior:
 *        c(k+1) − c(k)·S(k)  es una constante ≥ 0.
 * Entonces los rangos de cada término no se solapan y dos puntos distintos del dominio no pueden dar el mismo índice.
 * Los coeficientes y los tamaños son polinomios en los parámetros (N, N − 2, N·N…), así que `i·N + j` con `j < N`
 * se prueba sin conocer el valor de N.
 *
 * Es una condición SUFICIENTE, no necesaria: si no se puede probar, se rechaza (el kernel ingenuo secuencial es
 * siempre correcto, una carrera de datos no).
 */

export interface DimensionDominio {
  /** Nombre del iterador. */
  iter: string;
  /** Cantidad de valores que toma (iter va de 0 a size − 1). */
  size: Expr;
}

export type Resultado = { ok: true } | { ok: false; motivo: string };

// ---- polinomios con coeficientes enteros: monomio ("N*N", "" = constante) → coeficiente ----

type Poli = Map<string, number>;

const cte = (n: number): Poli => (n === 0 ? new Map() : new Map([["", n]]));
const variable = (v: string): Poli => new Map([[v, 1]]);

function limpiar(p: Poli): Poli {
  for (const [k, v] of p) if (v === 0) p.delete(k);
  return p;
}

function suma(a: Poli, b: Poli, signo: 1 | -1 = 1): Poli {
  const r = new Map(a);
  for (const [k, v] of b) r.set(k, (r.get(k) ?? 0) + signo * v);
  return limpiar(r);
}

function producto(a: Poli, b: Poli): Poli {
  const r: Poli = new Map();
  for (const [ka, va] of a) {
    for (const [kb, vb] of b) {
      const factores = [...(ka ? ka.split("*") : []), ...(kb ? kb.split("*") : [])].sort();
      const k = factores.join("*");
      r.set(k, (r.get(k) ?? 0) + va * vb);
    }
  }
  return limpiar(r);
}

/** Si el polinomio es una constante, devuelve su valor. */
const comoConstante = (p: Poli): number | undefined => {
  if (p.size === 0) return 0;
  if (p.size === 1 && p.has("")) return p.get("")!;
  return undefined;
};

class NoLineal extends Error {}

/** Convierte una expresión del frontend en un polinomio; lanza NoLineal si tiene algo que no es +, −, × entre números y variables. */
function aPolinomio(e: Expr, variablesPermitidas: (nombre: string) => boolean): Poli {
  switch (e.tipo) {
    case "num":
      if (!Number.isInteger(e.valor)) throw new NoLineal(`el número ${e.valor} no es entero`);
      return cte(e.valor);
    case "var":
      if (!variablesPermitidas(e.nombre)) throw new NoLineal(`«${e.nombre}» no es un iterador ni un parámetro (puede cambiar de valor)`);
      return variable(e.nombre);
    case "bin":
      if (e.op === "+") return suma(aPolinomio(e.izq, variablesPermitidas), aPolinomio(e.der, variablesPermitidas));
      if (e.op === "-") return suma(aPolinomio(e.izq, variablesPermitidas), aPolinomio(e.der, variablesPermitidas), -1);
      if (e.op === "*") return producto(aPolinomio(e.izq, variablesPermitidas), aPolinomio(e.der, variablesPermitidas));
      throw new NoLineal("contiene una división");
    case "acceso":
      throw new NoLineal(`depende de los datos de «${e.arreglo}»`);
    case "llamada":
      throw new NoLineal(`llama a ${e.funcion}`);
    case "desconocida":
      throw new NoLineal(`contiene «${e.texto}»`);
  }
}

/** Separa el polinomio en  Σ coef(iter)·iter  (+ un resto sin iteradores, que se ignora). */
function coeficientes(p: Poli, iteradores: string[]): Map<string, Poli> {
  const coefs = new Map<string, Poli>(iteradores.map((i) => [i, new Map()]));
  for (const [monomio, valor] of p) {
    const factores = monomio ? monomio.split("*") : [];
    const usados = factores.filter((f) => iteradores.includes(f));
    if (usados.length === 0) continue; // parte constante para todos los hilos
    if (usados.length > 1) throw new NoLineal(`multiplica iteradores entre sí (${monomio})`);
    const iter = usados[0]!;
    const resto = factores.filter((f, i) => !(f === iter && i === factores.indexOf(iter))).join("*");
    const actual = coefs.get(iter)!;
    actual.set(resto, (actual.get(resto) ?? 0) + valor);
  }
  for (const c of coefs.values()) limpiar(c);
  return coefs;
}

/** Deja el polinomio con signo positivo si todos sus términos tienen el mismo signo; undefined si el signo no se puede decidir. */
function positivo(p: Poli): Poli | undefined {
  const valores = [...p.values()];
  if (valores.every((v) => v > 0)) return p;
  if (valores.every((v) => v < 0)) return new Map([...p].map(([k, v]) => [k, -v]));
  return undefined;
}

function permutaciones<T>(xs: T[]): T[][] {
  if (xs.length <= 1) return [xs];
  return xs.flatMap((x, i) => permutaciones([...xs.slice(0, i), ...xs.slice(i + 1)]).map((r) => [x, ...r]));
}

const texto = (p: Poli): string => (p.size === 0 ? "0" : [...p].map(([k, v]) => `${v}${k ? "·" + k : ""}`).join(" + "));

/**
 * @param indice  expresión del índice de escritura.
 * @param dominio iteradores que varían entre las ejecuciones que escriben (los del grid y los bucles internos que rodean la escritura).
 * @param esParametro  true si el nombre es un parámetro del kernel (constante durante la ejecución).
 */
export function probarInyectivo(indice: Expr, dominio: DimensionDominio[], esParametro: (nombre: string) => boolean): Resultado {
  const nombres = dominio.map((d) => d.iter);
  try {
    const coefs = coeficientes(aPolinomio(indice, (n) => nombres.includes(n) || esParametro(n)), nombres);

    const dims: { iter: string; coef: Poli; size: Poli }[] = [];
    for (const d of dominio) {
      const bruto = coefs.get(d.iter)!;
      if (bruto.size === 0) return { ok: false, motivo: `el índice no depende de «${d.iter}», así que hilos que solo difieren en «${d.iter}» escriben la misma posición` };
      const coef = positivo(bruto);
      if (!coef) return { ok: false, motivo: `no se puede decidir el signo del paso de «${d.iter}» (${texto(bruto)})` };
      let size: Poli;
      try {
        size = aPolinomio(d.size, esParametro);
      } catch (e) {
        if (!(e instanceof NoLineal)) throw e;
        return { ok: false, motivo: `el tamaño de «${d.iter}» no es una expresión simple (${e.message})` };
      }
      dims.push({ iter: d.iter, coef, size });
    }

    // Algún orden de los iteradores forma una cadena de base mixta.
    for (const orden of permutaciones(dims)) {
      const primero = comoConstante(orden[0]!.coef);
      if (primero === undefined || primero <= 0) continue; // un paso simbólico podría valer 0 en tiempo de ejecución
      let cadena = true;
      for (let k = 1; k < orden.length && cadena; k++) {
        const prev = orden[k - 1]!;
        const holgura = comoConstante(suma(orden[k]!.coef, producto(prev.coef, prev.size), -1));
        cadena = holgura !== undefined && holgura >= 0;
      }
      if (cadena) return { ok: true };
    }

    const detalle = dims.map((d) => `paso de «${d.iter}» = ${texto(d.coef)}, tamaño = ${texto(d.size)}`).join("; ");
    return { ok: false, motivo: `no se puede probar que dos hilos distintos escriban posiciones distintas (${detalle})` };
  } catch (e) {
    if (e instanceof NoLineal) return { ok: false, motivo: `el índice no es una combinación lineal de los iteradores: ${e.message}` };
    throw e;
  }
}
