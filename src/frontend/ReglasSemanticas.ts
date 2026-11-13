import type { Expr, NodoBucle, Sentencia } from '../core/Modelo.js';
import type { IReglaSemantica, ResultadoReglaSemantica } from './IReglaSemantica.js';

const OK: ResultadoReglaSemantica = { ok: true };

function contieneExpresionDesconocida(expr: Expr): boolean {
    if (expr.tipo === 'desconocida') return true;
    if (expr.tipo === 'bin') return contieneExpresionDesconocida(expr.izq) || contieneExpresionDesconocida(expr.der);
    if (expr.tipo === 'acceso') return expr.indices.some(contieneExpresionDesconocida);
    if (expr.tipo === 'llamada') return expr.args.some(contieneExpresionDesconocida);
    return false;
}

function cuerpoReasignaIterador(sentencias: Sentencia[], iterador: string): boolean {
    for (const sentencia of sentencias) {
        if (sentencia.tipo === 'bucle' && cuerpoReasignaIterador(sentencia.bucle.cuerpo, iterador)) return true;
        if (sentencia.tipo !== 'asignacion') continue;
        if (sentencia.destino.tipo === 'var' && sentencia.destino.nombre === iterador) return true;
    }
    return false;
}

export class ValidacionTipos implements IReglaSemantica {
    readonly nombre = 'ValidacionTipos';

    validar(bucle: NodoBucle): ResultadoReglaSemantica {
        if (contieneExpresionDesconocida(bucle.inicio) || contieneExpresionDesconocida(bucle.limite) || contieneExpresionDesconocida(bucle.paso)) {
            return { ok: false, mensaje: 'la cabecera del bucle contiene expresiones no soportadas' };
        }
        return OK;
    }
}

export class ValidacionPureza implements IReglaSemantica {
    readonly nombre = 'ValidacionPureza';

    validar(bucle: NodoBucle): ResultadoReglaSemantica {
        if (cuerpoReasignaIterador(bucle.cuerpo, bucle.variable)) {
            return { ok: false, mensaje: `el iterador «${bucle.variable}» no debe reasignarse dentro del cuerpo` };
        }
        return OK;
    }
}
