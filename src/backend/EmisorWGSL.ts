import type { KernelIR, Expr, Stmt } from "../ir/ir-contract.js";
import { TablaSimbolos } from "./TablaSimbolos.js";

// Hilos por workgroup según dimensiones del grid (64 en total en 1D/2D, 64 en 3D).
const TAMANO_WORKGROUP: Record<number, number[]> = { 1: [64, 1, 1], 2: [8, 8, 1], 3: [4, 4, 4] };

export class EmisorWGSL {
  private tabla = new TablaSimbolos();
  private indentacion = 1;

  emitir(kernel: KernelIR): string {
    // 1. Poblar la tabla de símbolos con params, buffers y grid
    this.inicializarSimbolos(kernel);

    // 2. Generar el encabezado (structs y bindings)
    const encabezado = this.emitirEncabezado(kernel);

    // 3. Generar la función compute y workgroup_size
    const cuerpoFuncion = this.emitirCuerpo(kernel);

    return `${encabezado}\n${cuerpoFuncion}`;
  }

  private inicializarSimbolos(kernel: KernelIR): void {
    for (const p of kernel.params) {
      this.tabla.registrar(p.name, { tipo: p.type, origen: "param" });
    }
    for (const b of kernel.buffers) {
      this.tabla.registrar(b.name, { tipo: b.type, origen: "buffer", acceso: b.access });
    }
    for (const g of kernel.grid) {
      this.tabla.registrar(g.iter, { tipo: "i32", origen: "iter" });
    }
  }

  private emitirEncabezado(kernel: KernelIR): string {
    const lineas: string[] = [];

    // Params struct si existen
    if (kernel.params.length > 0) {
      lineas.push("struct Params {");
      for (const p of kernel.params) {
        lineas.push(`  ${p.name}: ${p.type},`);
      }
      lineas.push("};");
      lineas.push("@group(0) @binding(0) var<uniform> params: Params;");
    }

    // Buffers como storage
    let bindingIndex = kernel.params.length > 0 ? 1 : 0;
    for (const b of kernel.buffers) {
      const mode = b.access === "read" ? "read" : "read_write";
      lineas.push(
        `@group(0) @binding(${bindingIndex}) var<storage, ${mode}> ${b.name}: array<${b.type}>;`
      );
      bindingIndex++;
    }

    return lineas.join("\n");
  }

  private emitirCuerpo(kernel: KernelIR): string {
    const lineas: string[] = [];
    const workgroup = TAMANO_WORKGROUP[kernel.grid.length] ?? [64, 1, 1];

    lineas.push(`@compute @workgroup_size(${workgroup.join(", ")})`);
    lineas.push(`fn ${kernel.name}(@builtin(global_invocation_id) gid: vec3<u32>) {`);

    // Mapeo del grid a variables i32
    const ejes = ["x", "y", "z"];
    kernel.grid.forEach((g, index) => {
      lineas.push(`  let ${g.iter} = i32(gid.${ejes[index]});`);
    });

    // Guardas de seguridad para hilos excedentes
    const condicionesGuarda = kernel.grid.map(
      (g) => `${g.iter} >= ${this.emitirExpr(g.size)}`
    );
    if (condicionesGuarda.length > 0) {
      lineas.push(`  if (${condicionesGuarda.join(" || ")}) { return; }`);
    }

    // Emisión de sentencias del body
    for (const stmt of kernel.body) {
      lineas.push(this.emitirStmt(stmt));
    }

    lineas.push("}");
    return lineas.join("\n");
  }

  private emitirExpr(expr: Expr): string {
    switch (expr.op) {
      case "lit":
        return expr.type === "f32" && /^-?\d+$/.test(String(expr.value))
          ? `${expr.value}.0`
          : `${expr.value}`;

      case "ref": {
        const info = this.tabla.obtener(expr.id);
        return info.origen === "param" ? `params.${expr.id}` : expr.id;
      }

      case "read": {
        const buf = this.tabla.obtener(expr.buffer);
        if (buf.origen !== "buffer") throw new Error(`${expr.buffer} no es un buffer`);
        return `${expr.buffer}[${this.emitirExpr(expr.index)}]`;
      }

      case "bin":
        return `(${this.emitirExpr(expr.l)} ${expr.operator} ${this.emitirExpr(expr.r)})`;

      case "unary":
        return `(${expr.operator}${this.emitirExpr(expr.x)})`;

      case "cast":
        return `${expr.targetType}(${this.emitirExpr(expr.x)})`;

      case "call":
        return `${expr.func}(${expr.args.map((a) => this.emitirExpr(a)).join(", ")})`;

      case "select":
        return `select(${this.emitirExpr(expr.b)}, ${this.emitirExpr(expr.a)}, ${this.emitirExpr(expr.cond)})`;
    }
  }

  private emitirStmt(stmt: Stmt): string {
    const indent = "  ".repeat(this.indentacion);

    switch (stmt.op) {
      case "var":
        this.tabla.registrar(stmt.id, { tipo: stmt.type, origen: "var" });
        return `${indent}var ${stmt.id}: ${stmt.type} = ${this.emitirExpr(stmt.init)};`;

      case "set":
        this.tabla.obtener(stmt.id); // Valida que exista
        return `${indent}${stmt.id} = ${this.emitirExpr(stmt.value)};`;

      case "store": {
        const buf = this.tabla.obtener(stmt.buffer);
        if (buf.acceso === "read") {
          throw new Error(`Violación de acceso: escritura en buffer de solo lectura "${stmt.buffer}"`);
        }
        return `${indent}${stmt.buffer}[${this.emitirExpr(stmt.index)}] = ${this.emitirExpr(stmt.value)};`;
      }

      case "for": {
        this.tabla.registrar(stmt.iter, { tipo: "i32", origen: "iter" });
        const fromStr = this.emitirExpr(stmt.from);
        const toStr = this.emitirExpr(stmt.to);

        const lineas = [
          `${indent}for (var ${stmt.iter} = ${fromStr}; ${stmt.iter} < ${toStr}; ${stmt.iter}++) {`,
        ];

        this.indentacion++;
        for (const s of stmt.body) {
          lineas.push(this.emitirStmt(s));
        }
        this.indentacion--;

        lineas.push(`${indent}}`);
        this.tabla.eliminar(stmt.iter); // Sale del scope
        return lineas.join("\n");
      }
    }
  }
}