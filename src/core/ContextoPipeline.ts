import type { NodoBucle, Diagnostico } from './Modelo.js';

export class ContextoPipeline {
    buclesCandidatos: NodoBucle[] = [];
    buclesAptos: NodoBucle[] = [];
    diagnosticos: Diagnostico[] = [];

    constructor(public readonly archivosEntrada: string[]) { }
}