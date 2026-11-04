import type { Expr, KernelIR } from '../ir/ir-contract.js';

const ref = (id: string): Expr => ({ op: 'ref', id });
const mul = (l: Expr, r: Expr): Expr => ({ op: 'bin', operator: '*', l, r });
const sum = (l: Expr, r: Expr): Expr => ({ op: 'bin', operator: '+', l, r });
const entero = (value: number): Expr => ({ op: 'lit', type: 'i32', value });

/**
 * out[<indice>] = f32(i * 1000 + j), con un grid no cuadrado: i < R, j < C.
 * `filaMayor` indexa i * C + j (j contiguo); si no, j * R + i (i contiguo).
 */
export function kernelNoCuadrado(filaMayor: boolean): KernelIR {
    const indice = filaMayor ? sum(mul(ref('i'), ref('C')), ref('j')) : sum(mul(ref('j'), ref('R')), ref('i'));
    return {
        irVersion: 1,
        name: filaMayor ? 'fila_mayor' : 'columna_mayor',
        params: [{ name: 'R', type: 'i32' }, { name: 'C', type: 'i32' }],
        buffers: [{ name: 'out', type: 'f32', access: 'write' }],
        grid: [{ iter: 'i', size: ref('R') }, { iter: 'j', size: ref('C') }],
        body: [{ op: 'store', buffer: 'out', index: indice, value: { op: 'cast', targetType: 'f32', x: sum(mul(ref('i'), entero(1000)), ref('j')) } }],
    };
}

/**
 * C[M×P] = A[M×K] · B[K×P] con las tres dimensiones distintas. Los pasos de los índices son parámetros (K y P),
 * así que una baldosa mal calculada se nota: con matrices cuadradas varios errores de índice se cancelan.
 */
export function kernelMatmulRectangular(): KernelIR {
    return {
        irVersion: 1,
        name: 'matmul_rect',
        params: [{ name: 'M', type: 'i32' }, { name: 'P', type: 'i32' }, { name: 'K', type: 'i32' }],
        buffers: [
            { name: 'A', type: 'f32', access: 'read' },
            { name: 'B', type: 'f32', access: 'read' },
            { name: 'C', type: 'f32', access: 'write' },
        ],
        grid: [{ iter: 'i', size: ref('M') }, { iter: 'j', size: ref('P') }],
        body: [
            { op: 'var', id: 'acc', type: 'f32', init: { op: 'lit', type: 'f32', value: 0 } },
            {
                op: 'for', iter: 'k', from: entero(0), to: ref('K'),
                body: [{
                    op: 'set', id: 'acc',
                    value: { op: 'bin', operator: '+', l: ref('acc'), r: { op: 'bin', operator: '*',
                        l: { op: 'read', buffer: 'A', index: sum(mul(ref('i'), ref('K')), ref('k')) },
                        r: { op: 'read', buffer: 'B', index: sum(mul(ref('k'), ref('P')), ref('j')) } } },
                }],
            },
            { op: 'store', buffer: 'C', index: sum(mul(ref('i'), ref('P')), ref('j')), value: ref('acc') },
        ],
    };
}
