import type { ContextoPipeline } from "./ContextoPipeline.js";
import type { IFase } from "./IFase.js";

export class TranspiladorPipeline {
    constructor(private fases: IFase[]) { }

    ejecutar(ctx: ContextoPipeline): ContextoPipeline {
        let contextoActual = ctx;
        for (const fase of this.fases) {
            contextoActual = fase.ejecutar(contextoActual);
        }
        return contextoActual;
    }
}