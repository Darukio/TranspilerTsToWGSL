import type { IFase } from "../core/IFase.js";
import type { ContextoPipeline } from "../core/ContextoPipeline.js";
import { KernelIRSchema } from "../ir/ir-contract.js";
import { EmisorWGSL } from "./EmisorWGSL.js";

export class FaseBackEnd implements IFase {
  ejecutar(ctx: ContextoPipeline): ContextoPipeline {
    for (const irRaw of ctx.kernels) {
      const kernel = KernelIRSchema.parse(irRaw);
      try {
        const emisor = new EmisorWGSL();
        const wgsl = emisor.emitir(kernel);
        ctx.shaders.push({ nombre: kernel.name, wgsl, optimizaciones: emisor.optimizaciones });
      } catch (error) {
        ctx.diagnosticos.push({
          fase: "backend",
          mensaje: `Kernel "${kernel.name}": ${error instanceof Error ? error.message : String(error)}`,
        });
      }
    }
    return ctx;
  }
}
