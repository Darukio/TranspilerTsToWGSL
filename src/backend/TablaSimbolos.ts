import type { DataType, AccessMode } from "../ir/ir-contract.js";

export interface SimboloInfo {
  tipo: DataType;
  origen: "param" | "buffer" | "iter" | "var";
  acceso?: AccessMode;
}

// Palabras clave y tipos de WGSL, más nombres internos que usa el emisor (params, gid).
const RESERVADAS = new Set([
  "alias", "break", "case", "const", "const_assert", "continue", "continuing", "default",
  "diagnostic", "discard", "else", "enable", "false", "fn", "for", "if", "let", "loop",
  "override", "requires", "return", "struct", "switch", "true", "var", "while",
  "array", "atomic", "bool", "f16", "f32", "i32", "u32", "ptr", "sampler", "select",
  "vec2", "vec3", "vec4", "mat2x2", "mat3x3", "mat4x4",
  "params", "gid",
]);

export class TablaSimbolos {
  private simbolos = new Map<string, SimboloInfo>();

  registrar(nombre: string, info: SimboloInfo): void {
    if (RESERVADAS.has(nombre)) {
      throw new Error(`Identificador reservado de WGSL: "${nombre}"`);
    }
    if (this.simbolos.has(nombre)) {
      throw new Error(`Símbolo ya declarado: "${nombre}"`);
    }
    this.simbolos.set(nombre, info);
  }

  obtener(nombre: string): SimboloInfo {
    const s = this.simbolos.get(nombre);
    if (!s) {
      throw new Error(`Símbolo no definido: "${nombre}"`);
    }
    return s;
  }

  eliminar(nombre: string): void {
    this.simbolos.delete(nombre);
  }
}
