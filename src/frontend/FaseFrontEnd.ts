import type { IFase } from "../core/IFase.js";
import type { ContextoPipeline } from "../core/ContextoPipeline.js";
import type { IGestorAST } from "../core/IGestorAST.js";

export class FaseFrontEnd implements IFase {
    constructor(private readonly gestor: IGestorAST) { }

    ejecutar(ctx: ContextoPipeline): ContextoPipeline {
        this.gestor.cargar(ctx.archivosEntrada);
        ctx.buclesCandidatos = this.gestor.obtenerBuclesCandidatos();
        ctx.diagnosticos.push(...this.gestor.obtenerDiagnosticos());
        return ctx;
    }
}