import type { NodoBucle, Diagnostico } from './Modelo.js';

export interface IGestorAST {
    cargar(archivos: string[]): void;
    obtenerBuclesCandidatos(): NodoBucle[];
    obtenerDiagnosticos(): Diagnostico[];
    obtenerTipoVariable(bucleId: string, nombreVar: string): string | undefined;
}