import { ContextoPipeline } from "./ContextoPipeline.js";
import { PipelineBuilder } from "./PipelineBuilder.js";
import { FaseFrontEnd } from "../frontend/FaseFrontEnd.js";
import { TSCompilerFacade } from "../frontend/TSCompilerFacade.js";
import { FaseMiddleEnd } from "../middleend/FaseMiddleEnd.js";
import { FaseBackEnd } from "../backend/FaseBackend.js";

export class TranspiladorTsaWGSL {
    transpilar(archivos: string[]): ContextoPipeline {
        const pipeline = new PipelineBuilder()
            .addFase(new FaseFrontEnd(new TSCompilerFacade()))
            .addFase(new FaseMiddleEnd())
            .addFase(new FaseBackEnd())
            .build();

        return pipeline.ejecutar(new ContextoPipeline(archivos));
    }
}