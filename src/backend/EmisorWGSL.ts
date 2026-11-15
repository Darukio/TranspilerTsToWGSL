import type { KernelIR, Expr, Stmt } from "../ir/ir-contract.js";
import { TablaSimbolos } from "./TablaSimbolos.js";
import { asignarEjes, calcularBindings, tamanoWorkgroup } from "./Bindings.js";
import { BALDOSA, detectarBaldosas, sustituir, type PlanBaldosas } from "./Baldosas.js";

export interface OpcionesEmisor {
  /** Emitir el kernel con baldosas en memoria compartida cuando el IR es una multiplicación de matrices (por defecto, sí). */
  baldosas?: boolean;
}

export class EmisorWGSL {
  private tabla = new TablaSimbolos();
  private indentacion = 1;
  /** Optimizaciones aplicadas al último kernel emitido (para informarlas). */
  optimizaciones: string[] = [];

  constructor(private readonly opciones: OpcionesEmisor = {}) { }

  emitir(kernel: KernelIR): string {
    this.optimizaciones = [];
    // 1. Poblar la tabla de símbolos con params, buffers y grid
    this.inicializarSimbolos(kernel);

    // 2. Generar el encabezado (structs y bindings)
    const encabezado = this.emitirEncabezado(kernel);

    // 3. Generar la función compute y workgroup_size (con baldosas si el kernel es una multiplicación de matrices)
    const plan = this.opciones.baldosas === false ? undefined : detectarBaldosas(kernel);
    if (plan) this.optimizaciones.push(`baldosas ${BALDOSA}×${BALDOSA} en memoria compartida`);
    const cuerpoFuncion = plan ? this.emitirCuerpoConBaldosas(kernel, plan) : this.emitirCuerpo(kernel);

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

    const bindings = calcularBindings(kernel);

    // Params struct si existen
    if (bindings.params) {
      lineas.push("struct Params {");
      for (const p of bindings.params.campos) {
        lineas.push(`  ${p.name}: ${p.type},`);
      }
      lineas.push("};");
      lineas.push(`@group(0) @binding(${bindings.params.binding}) var<uniform> params: Params;`);
    }

    // Buffers como storage
    for (const b of bindings.buffers) {
      const mode = b.access === "read" ? "read" : "read_write";
      lineas.push(
        `@group(0) @binding(${b.binding}) var<storage, ${mode}> ${b.name}: array<${b.type}>;`
      );
    }

    return lineas.join("\n");
  }

  /**
   * Multiplicación de matrices con baldosas: cada workgroup carga una baldosa de A y otra de B en memoria compartida,
   * todos los hilos las reutilizan BALDOSA veces y recién entonces se pasa a la siguiente. Se lee de memoria global
   * ~BALDOSA veces menos que en el kernel ingenuo.
   *
   * Reglas de WGSL que condicionan la forma: workgroupBarrier() exige que TODOS los hilos del grupo lleguen a ella,
   * por eso no hay `return` temprano (las lecturas y la escritura final se protegen con guardas) y la cota del bucle
   * es uniforme (solo depende de parámetros). Fuera de rango se carga 0.0, que no altera la suma.
   */
  private emitirCuerpoConBaldosas(kernel: KernelIR, plan: PlanBaldosas): string {
    const T = BALDOSA;
    const idxLocal = (fila: string, col: string) => `${fila} * ${T} + ${col}`;
    const lineas: string[] = [];

    // Índices de las lecturas con el k de cada baldosa (ka_ para A, kb_ para B).
    this.tabla.abrirAmbito();
    this.tabla.registrar("ka_", { tipo: "i32", origen: "iter" });
    this.tabla.registrar("kb_", { tipo: "i32", origen: "iter" });
    const indiceA = this.emitirExpr(sustituir(plan.a.indice, plan.k.iter, "ka_"));
    const indiceB = this.emitirExpr(sustituir(plan.b.indice, plan.k.iter, "kb_"));
    this.tabla.cerrarAmbito();

    const tamFila = this.emitirExpr(plan.fila.size);
    const tamCol = this.emitirExpr(plan.col.size);
    const cota = this.emitirExpr(plan.k.hasta);
    const inicial = this.emitirExpr(plan.acumulador.init);
    this.tabla.registrar(plan.acumulador.id, { tipo: "f32", origen: "var" });
    const indiceC = this.emitirExpr(plan.salida.indice);

    lineas.push(`var<workgroup> tileA_: array<f32, ${T * T}>;`);
    lineas.push(`var<workgroup> tileB_: array<f32, ${T * T}>;`);
    lineas.push(`@compute @workgroup_size(${T}, ${T}, 1)`);
    lineas.push(`fn ${kernel.name}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {`);
    // x recorre las columnas (índice contiguo de B y C) e y las filas; detectarBaldosas exige que asignarEjes coincida.
    lineas.push(`  let ${plan.col.iter} = i32(gid.x);`);
    lineas.push(`  let ${plan.fila.iter} = i32(gid.y);`);
    lineas.push(`  let lx_ = i32(lid.x);`);
    lineas.push(`  let ly_ = i32(lid.y);`);
    lineas.push(`  let filaOk_ = ${plan.fila.iter} < ${tamFila};`);
    lineas.push(`  let colOk_ = ${plan.col.iter} < ${tamCol};`);
    lineas.push(`  let K_ = ${cota};`);
    lineas.push(`  var ${plan.acumulador.id}: f32 = ${inicial};`);
    lineas.push(`  for (var kk_ = 0; kk_ < K_; kk_ += ${T}) {`);
    lineas.push(`    let ka_ = kk_ + lx_;`);
    lineas.push(`    let kb_ = kk_ + ly_;`);
    lineas.push(`    if (filaOk_ && ka_ < K_) { tileA_[${idxLocal("ly_", "lx_")}] = ${plan.a.buffer}[${indiceA}]; } else { tileA_[${idxLocal("ly_", "lx_")}] = 0.0; }`);
    lineas.push(`    if (kb_ < K_ && colOk_) { tileB_[${idxLocal("ly_", "lx_")}] = ${plan.b.buffer}[${indiceB}]; } else { tileB_[${idxLocal("ly_", "lx_")}] = 0.0; }`);
    lineas.push(`    workgroupBarrier();`);
    lineas.push(`    for (var t_ = 0; t_ < ${T}; t_++) { ${plan.acumulador.id} += tileA_[${idxLocal("ly_", "t_")}] * tileB_[${idxLocal("t_", "lx_")}]; }`);
    lineas.push(`    workgroupBarrier();`);
    lineas.push(`  }`);
    lineas.push(`  if (filaOk_ && colOk_) { ${plan.salida.buffer}[${indiceC}] = ${plan.acumulador.id}; }`);
    lineas.push(`}`);
    return lineas.join("\n");
  }

  private emitirCuerpo(kernel: KernelIR): string {
    const lineas: string[] = [];
    const workgroup = tamanoWorkgroup(kernel.grid.length);

    lineas.push(`@compute @workgroup_size(${workgroup.join(", ")})`);
    lineas.push(`fn ${kernel.name}(@builtin(global_invocation_id) gid: vec3<u32>) {`);

    // Mapeo del grid a variables i32: el eje x va al iterador de paso 1 en memoria (ver asignarEjes)
    const ejes = ["x", "y", "z"];
    asignarEjes(kernel).forEach((indiceGrid, eje) => {
      lineas.push(`  let ${kernel.grid[indiceGrid]!.iter} = i32(gid.${ejes[eje]});`);
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
        this.tabla.abrirAmbito();
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
        this.tabla.cerrarAmbito(); // Sale del scope: se olvidan el iterador y las variables del cuerpo
        return lineas.join("\n");
      }
    }
  }
}