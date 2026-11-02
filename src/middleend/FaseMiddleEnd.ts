import type { IFase } from "../core/IFase.js";
import type { ContextoPipeline } from "../core/ContextoPipeline.js";
import type { Expr as ExprFE, NodoBucle, Sentencia } from "../core/Modelo.js";
import type { DataType, Expr, Stmt, KernelIR, AccessMode } from "../ir/ir-contract.js";

class NoApto extends Error { }

const FUNCIONES_MATH: Record<string, string> = {
    "Math.sqrt": "sqrt", "Math.sin": "sin", "Math.cos": "cos", "Math.tan": "tan",
    "Math.abs": "abs", "Math.exp": "exp", "Math.log": "log", "Math.floor": "floor",
    "Math.ceil": "ceil", "Math.pow": "pow", "Math.min": "min", "Math.max": "max",
};

interface Tipado { e: Expr; t: DataType }

/** Convierte bucles candidatos del frontend en KernelIR, descartando los que no son paralelizables. */
export class FaseMiddleEnd implements IFase {
    ejecutar(ctx: ContextoPipeline): ContextoPipeline {
        for (const bucle of ctx.buclesCandidatos) {
            try {
                ctx.kernels.push(new Reductor(bucle).reducir());
                ctx.buclesAptos.push(bucle);
            } catch (error) {
                if (!(error instanceof NoApto)) throw error;
                ctx.diagnosticos.push({ fase: "middleend", mensaje: `${bucle.id}: ${error.message}`, ubicacion: bucle.ubicacion });
            }
        }
        return ctx;
    }
}

class Reductor {
    private grid: { iter: string; size: ExprFE }[] = [];
    private cuerpo: Sentencia[] = [];
    private iteradores = new Set<string>();
    private buffers = new Map<string, { leido: boolean; escrito: boolean }>();
    private intVars = new Set<string>();
    private params = new Map<string, DataType>();

    constructor(private readonly raiz: NodoBucle) { }

    reducir(): KernelIR {
        this.separarGrid();
        this.analizar(this.cuerpo);
        this.verificarDependencias();
        const body = this.cuerpo.map(s => this.sentencia(s));
        const gridIR = this.grid.map(g => ({ iter: g.iter, size: this.coercer(this.expr(g.size, "i32"), "i32") }));

        return {
            irVersion: 1,
            name: `kernel_${this.raiz.id}`,
            params: [...this.params].map(([name, type]) => ({ name, type })),
            buffers: [...this.buffers].map(([name, u]) => ({
                name, type: "f32" as const,
                access: (u.leido && u.escrito ? "readwrite" : u.escrito ? "write" : "read") as AccessMode,
            })),
            grid: gridIR,
            body,
        };
    }

    // ---- 1. Qué bucles forman el grid de hilos ----
    private separarGrid(): void {
        let actual: NodoBucle = this.raiz;
        for (;;) {
            if (!this.esDimensionGrid(actual)) {
                if (this.grid.length === 0) throw new NoApto("el bucle externo debe iterar desde 0 con paso 1");
                break;
            }
            this.grid.push({ iter: actual.variable, size: this.limiteExclusivo(actual) });
            const unico = actual.cuerpo.length === 1 ? actual.cuerpo[0] : undefined;
            if (this.grid.length < 3 && unico?.tipo === "bucle" && this.independiente(unico.bucle)) {
                actual = unico.bucle;
                continue;
            }
            this.cuerpo = actual.cuerpo;
            return;
        }
        // El último candidato a dimensión no cumplía: queda como `for` interno.
        this.cuerpo = [{ tipo: "bucle", bucle: actual }];
    }

    private esDimensionGrid(b: NodoBucle): boolean {
        return b.inicio.tipo === "num" && b.inicio.valor === 0 && b.paso.tipo === "num" && b.paso.valor === 1;
    }

    private independiente(b: NodoBucle): boolean {
        const previos = new Set(this.grid.map(g => g.iter));
        return ![b.inicio, b.limite].some(e => [...variables(e)].some(v => previos.has(v)));
    }

    private limiteExclusivo(b: NodoBucle): ExprFE {
        return b.inclusivo ? { tipo: "bin", op: "+", izq: b.limite, der: { tipo: "num", valor: 1 } } : b.limite;
    }

    // ---- 2. Análisis: sentencias soportadas, uso de buffers, variables enteras ----
    private analizar(sents: Sentencia[]): void {
        for (const s of sents) {
            if (s.tipo === "desconocida") throw new NoApto(`sentencia no soportada: ${s.texto}`);
            if (s.tipo === "bucle") {
                const b = s.bucle;
                if (b.paso.tipo !== "num" || b.paso.valor !== 1) throw new NoApto("solo se soportan bucles internos con paso 1");
                this.iteradores.add(b.variable);
                this.recorrer(b.inicio, true, false);
                this.recorrer(b.limite, true, false);
                this.analizar(b.cuerpo);
                continue;
            }
            if (s.destino.tipo !== "acceso") throw new NoApto("solo se soportan escrituras a elementos de arreglo (una escritura a escalar es una carrera entre hilos)");
            this.recorrer(s.destino, false, true);
            this.recorrer(s.valor, false, false);
        }
        for (const g of this.grid) this.iteradores.add(g.iter);
        for (const g of this.grid) this.recorrer(g.size, true, false);
    }

    private recorrer(e: ExprFE, entero: boolean, escritura: boolean): void {
        switch (e.tipo) {
            case "num": return;
            case "var": if (entero) this.intVars.add(e.nombre); return;
            case "bin": this.recorrer(e.izq, entero, false); this.recorrer(e.der, entero, false); return;
            case "llamada":
                if (!(e.funcion in FUNCIONES_MATH)) throw new NoApto(`función no soportada: ${e.funcion}`);
                e.args.forEach(a => this.recorrer(a, false, false));
                return;
            case "acceso": {
                if (e.indices.length !== 1) throw new NoApto(`indexado multidimensional no soportado en "${e.arreglo}" (use índice lineal)`);
                const uso = this.buffers.get(e.arreglo) ?? { leido: false, escrito: false };
                if (escritura) uso.escrito = true; else uso.leido = true;
                this.buffers.set(e.arreglo, uso);
                this.recorrer(e.indices[0]!, true, false);
                return;
            }
            case "desconocida": throw new NoApto(`expresión no soportada: ${e.texto}`);
        }
    }

    // ---- 3. Dependencias entre iteraciones (heurística conservadora) ----
    private verificarDependencias(): void {
        const escrituras = new Map<string, string>(); // arreglo -> índice escrito
        const lecturas: { arreglo: string; indice: string }[] = [];
        const visitarExpr = (e: ExprFE): void => {
            if (e.tipo === "acceso") { lecturas.push({ arreglo: e.arreglo, indice: clave(e.indices[0]!) }); e.indices.forEach(visitarExpr); }
            else if (e.tipo === "bin") { visitarExpr(e.izq); visitarExpr(e.der); }
            else if (e.tipo === "llamada") e.args.forEach(visitarExpr);
        };
        const visitar = (sents: Sentencia[]): void => {
            for (const s of sents) {
                if (s.tipo === "bucle") { visitar(s.bucle.cuerpo); continue; }
                if (s.tipo !== "asignacion" || s.destino.tipo !== "acceso") continue;
                const indice = clave(s.destino.indices[0]!);
                const previo = escrituras.get(s.destino.arreglo);
                if (previo !== undefined && previo !== indice) throw new NoApto(`"${s.destino.arreglo}" se escribe con distintos índices`);
                escrituras.set(s.destino.arreglo, indice);
                const vars = variables(s.destino.indices[0]!);
                for (const g of this.grid) {
                    if (!vars.has(g.iter)) throw new NoApto(`la escritura a "${s.destino.arreglo}" no depende de "${g.iter}": carrera entre hilos`);
                }
                visitarExpr(s.valor);
                s.destino.indices.forEach(visitarExpr);
            }
        };
        visitar(this.cuerpo);
        for (const l of lecturas) {
            const w = escrituras.get(l.arreglo);
            if (w !== undefined && w !== l.indice) throw new NoApto(`dependencia entre iteraciones: "${l.arreglo}" se lee y escribe en índices distintos`);
        }
    }

    // ---- 4. Lowering a IR ----
    private sentencia(s: Sentencia): Stmt {
        if (s.tipo === "bucle") {
            const b = s.bucle;
            return {
                op: "for", iter: b.variable,
                from: this.coercer(this.expr(b.inicio, "i32"), "i32"),
                to: this.coercer(this.expr(this.limiteExclusivo(b), "i32"), "i32"),
                body: b.cuerpo.map(x => this.sentencia(x)),
            };
        }
        if (s.tipo !== "asignacion" || s.destino.tipo !== "acceso") throw new NoApto("sentencia no soportada");
        return {
            op: "store", buffer: s.destino.arreglo,
            index: this.coercer(this.expr(s.destino.indices[0]!, "i32"), "i32"),
            value: this.coercer(this.expr(s.valor, "f32"), "f32"),
        };
    }

    private expr(e: ExprFE, quiere: DataType): Tipado {
        switch (e.tipo) {
            case "num":
                return quiere === "i32" && Number.isInteger(e.valor)
                    ? { e: { op: "lit", type: "i32", value: e.valor }, t: "i32" }
                    : { e: { op: "lit", type: "f32", value: e.valor }, t: "f32" };
            case "var": {
                if (this.iteradores.has(e.nombre)) return { e: { op: "ref", id: e.nombre }, t: "i32" };
                const t: DataType = this.intVars.has(e.nombre) ? "i32" : "f32";
                this.params.set(e.nombre, t);
                return { e: { op: "ref", id: e.nombre }, t };
            }
            case "bin": {
                let l = this.expr(e.izq, quiere), r = this.expr(e.der, quiere);
                if (l.t !== r.t) { l = this.aTipo(l, "f32"); r = this.aTipo(r, "f32"); }
                return { e: { op: "bin", operator: e.op, l: l.e, r: r.e }, t: l.t };
            }
            case "acceso":
                return { e: { op: "read", buffer: e.arreglo, index: this.coercer(this.expr(e.indices[0]!, "i32"), "i32") }, t: "f32" };
            case "llamada": {
                const func = FUNCIONES_MATH[e.funcion]!;
                const args = e.args.map(a => this.coercer(this.expr(a, "f32"), "f32"));
                return { e: { op: "call", func, args }, t: "f32" };
            }
            case "desconocida": throw new NoApto(`expresión no soportada: ${e.texto}`);
        }
    }

    private aTipo(x: Tipado, t: DataType): Tipado {
        return x.t === t ? x : { e: { op: "cast", targetType: t, x: x.e }, t };
    }

    private coercer(x: Tipado, t: DataType): Expr {
        return this.aTipo(x, t).e;
    }
}

function variables(e: ExprFE, acc = new Set<string>()): Set<string> {
    if (e.tipo === "var") acc.add(e.nombre);
    else if (e.tipo === "bin") { variables(e.izq, acc); variables(e.der, acc); }
    else if (e.tipo === "acceso") e.indices.forEach(i => variables(i, acc));
    else if (e.tipo === "llamada") e.args.forEach(a => variables(a, acc));
    return acc;
}

const clave = (e: ExprFE): string => JSON.stringify(e);
