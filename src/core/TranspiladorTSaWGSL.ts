import { ContextoPipeline } from "./ContextoPipeline.js";
import { PipelineBuilder } from "./PipelineBuilder.js";
import { FaseFrontEnd } from "../frontend/FaseFrontEnd.js";
import { TSCompilerFacade } from "../frontend/TSCompilerFacade.js";

export class TranspiladorTsaWGSL {
    transpilar(archivos: string[]): ContextoPipeline {
        const pipeline = new PipelineBuilder()
            .addFase(new FaseFrontEnd(new TSCompilerFacade()))
            .build();

        return pipeline.ejecutar(new ContextoPipeline(archivos));
    }
}