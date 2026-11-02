import type { ContextoPipeline } from './ContextoPipeline.js';

export interface IFase {
    ejecutar(ctx: ContextoPipeline): ContextoPipeline;
}