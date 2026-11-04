import type { NodoBucle, Diagnostico } from './Modelo.js';
import type { KernelIR } from '../ir/ir-contract.js';

export interface Shader {
    nombre: string;
    wgsl: string;
    /** Optimizaciones que el emisor aplicó a este kernel (vacío = kernel ingenuo). */
    optimizaciones: string[];
}

export class ContextoPipeline {
    buclesCandidatos: NodoBucle[] = [];
    buclesAptos: NodoBucle[] = [];
    kernels: KernelIR[] = [];
    shaders: Shader[] = [];
    diagnosticos: Diagnostico[] = [];

    constructor(public readonly archivosEntrada: string[]) { }
}
