import type { NodoBucle } from '../core/Modelo.js';

export type ResultadoReglaSemantica =
    | { ok: true }
    | { ok: false; mensaje: string };

export interface IReglaSemantica {
    readonly nombre: string;
    validar(bucle: NodoBucle): ResultadoReglaSemantica;
}
