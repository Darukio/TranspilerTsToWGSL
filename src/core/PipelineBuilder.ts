import type { IFase } from "./IFase.js";
import { TranspiladorPipeline } from "./TranspiladorPipeline.js";

export class PipelineBuilder {
    private fases: IFase[] = [];

    addFase(fase: IFase): PipelineBuilder {
        this.fases.push(fase);
        return this;
    }

    build(): TranspiladorPipeline {
        return new TranspiladorPipeline(this.fases);
    }
}