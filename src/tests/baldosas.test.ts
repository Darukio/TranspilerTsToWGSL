import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { TranspiladorTsaWGSL } from '../core/TranspiladorTSaWGSL.js';
import { EmisorWGSL } from '../backend/EmisorWGSL.js';
import { BALDOSA, detectarBaldosas } from '../backend/Baldosas.js';
import { crearBuffers } from '../runtime/buffers.js';
import { calcularWorkgroups, ejecutarKernel, leerWorkgroup } from '../runtime/dispatch.js';
import { leerSalidas } from '../runtime/lectura.js';
import type { Expr, KernelIR } from '../ir/ir-contract.js';
import { kernelMatmulRectangular, kernelNoCuadrado } from './ejemplos-ir.js';
import { conGPU } from './ayudas-gpu.js';

const kernelDe = (archivo: string): KernelIR => {
    const ctx = new TranspiladorTsaWGSL().transpilar([fileURLToPath(new URL(`../../src/pruebas/${archivo}.ts`, import.meta.url))]);
    assert.deepEqual(ctx.diagnosticos, []);
    return ctx.kernels[0]!;
};
const copia = (k: KernelIR): KernelIR => structuredClone(k);

test('detectarBaldosas reconoce la multiplicación de matrices (la transpilada y la rectangular)', () => {
    const plan = detectarBaldosas(kernelDe('matmul'))!;
    assert.deepEqual([plan.fila.iter, plan.col.iter, plan.k.iter, plan.acumulador.id, plan.a.buffer, plan.b.buffer, plan.salida.buffer], ['i', 'j', 'k', 'suma', 'A', 'B', 'C']);
    const rect = detectarBaldosas(kernelMatmulRectangular())!;
    assert.deepEqual([rect.fila.iter, rect.col.iter, rect.a.buffer, rect.b.buffer], ['i', 'j', 'A', 'B']);
});

test('detectarBaldosas no toca nada que no sea exactamente ese patrón', () => {
    for (const otro of ['vectores', 'matrices', 'monte-carlo']) assert.equal(detectarBaldosas(kernelDe(otro)), undefined, otro);
    assert.equal(detectarBaldosas(kernelNoCuadrado(true)), undefined);

    // Variantes inválidas de un kernel válido: cualquier duda → no se optimiza
    const sinCambios = kernelMatmulRectangular();
    const recorrer = (k: KernelIR) => (k.body[1] as Extract<KernelIR['body'][number], { op: 'for' }>);
    const casos: Record<string, (k: KernelIR) => void> = {
        'un buffer de entrada que también se escribe': (k) => { k.buffers[0]!.access = 'readwrite'; },
        'acumulador entero': (k) => { (k.body[0] as { type: string }).type = 'i32'; },
        'la cota del bucle depende de una lectura': (k) => { recorrer(k).to = { op: 'read', buffer: 'A', index: { op: 'lit', type: 'i32', value: 0 } }; },
        'el bucle no empieza en 0': (k) => { recorrer(k).from = { op: 'lit', type: 'i32', value: 1 }; },
        'sentencias extra en el cuerpo': (k) => { k.body.push({ op: 'store', buffer: 'C', index: { op: 'lit', type: 'i32', value: 0 }, value: { op: 'lit', type: 'f32', value: 0 } }); },
        'el paso del índice depende de un iterador': (k) => {
            const set = recorrer(k).body[0] as { value: { r: { l: { index: { l: { r: Expr } } } } } };
            set.value.r.l.index.l.r = { op: 'ref', id: 'j' };
        },
        'un símbolo choca con un nombre interno': (k) => { k.buffers[2]!.name = 'tileA_'; (k.body[2] as { buffer: string }).buffer = 'tileA_'; },
    };
    for (const [nombre, romper] of Object.entries(casos)) {
        const k = copia(sinCambios);
        romper(k);
        assert.equal(detectarBaldosas(k), undefined, nombre);
    }
    assert.notEqual(detectarBaldosas(sinCambios), undefined, 'el original sigue siendo válido');
});

test('el operando de la multiplicación puede venir en cualquier orden', () => {
    const k = kernelMatmulRectangular();
    const prod = ((k.body[1] as { body: { value: { r: { l: Expr; r: Expr } } }[] }).body[0]!).value.r;
    [prod.l, prod.r] = [prod.r, prod.l];
    assert.deepEqual(detectarBaldosas(k)?.a.buffer, 'A');
});

test('el WGSL con baldosas respeta las reglas de las barreras: sin return temprano, con guardas', () => {
    const wgsl = new EmisorWGSL().emitir(kernelMatmulRectangular());
    assert.match(wgsl, new RegExp(`@workgroup_size\\(${BALDOSA}, ${BALDOSA}, 1\\)`));
    assert.match(wgsl, /var<workgroup> tileA_: array<f32, 256>;/);
    assert.match(wgsl, /var<workgroup> tileB_: array<f32, 256>;/);
    assert.equal(wgsl.match(/workgroupBarrier\(\)/g)?.length, 2);
    assert.doesNotMatch(wgsl, /\breturn\b/, 'una barrera no puede tener hilos que ya salieron');
    assert.match(wgsl, /if \(filaOk_ && colOk_\) \{ C\[/);
    // el kernel ingenuo conserva el return temprano y el workgroup de 8×8
    const ingenuo = new EmisorWGSL({ baldosas: false }).emitir(kernelMatmulRectangular());
    assert.match(ingenuo, /@workgroup_size\(8, 8, 1\)/);
    assert.match(ingenuo, /return;/);
    assert.doesNotMatch(ingenuo, /workgroupBarrier/);
});

test('el runtime lee el tamaño de workgroup del propio shader, así no se puede desincronizar', () => {
    const k = kernelMatmulRectangular();
    assert.deepEqual(leerWorkgroup(new EmisorWGSL().emitir(k)), [16, 16, 1]);
    assert.deepEqual(leerWorkgroup(new EmisorWGSL({ baldosas: false }).emitir(k)), [8, 8, 1]);
    // 35 columnas y 19 filas con baldosas de 16: x cubre las columnas (3 grupos), y las filas (2 grupos)
    assert.deepEqual(calcularWorkgroups(k, { M: 19, P: 35, K: 21 }, [16, 16, 1]).grupos, [3, 2, 1]);
});

const productoCPU = (A: Float32Array, B: Float32Array, M: number, P: number, K: number) => {
    const C = new Float64Array(M * P);
    for (let i = 0; i < M; i++) for (let j = 0; j < P; j++) { let s = 0; for (let k = 0; k < K; k++) s += A[i * K + k]! * B[k * P + j]!; C[i * P + j] = s; }
    return C;
};
const cercanos = (real: ArrayLike<number>, esperado: ArrayLike<number>, tol: number, donde: string) => {
    assert.equal(real.length, esperado.length, donde);
    for (let i = 0; i < esperado.length; i++) assert.ok(Math.abs(real[i]! - esperado[i]!) <= tol * Math.max(1, Math.abs(esperado[i]!)), `${donde} [${i}]: GPU=${real[i]} CPU=${esperado[i]}`);
};

for (const [M, P, K] of [[1, 1, 1], [15, 15, 15], [16, 16, 16], [17, 17, 17], [19, 35, 21], [33, 7, 50], [64, 64, 64]] as const) {
    test(`matmul con baldosas en la GPU, ${M}×${K} · ${K}×${P}: coincide con la CPU y con el kernel ingenuo`, async (t) => {
        const kernel = kernelMatmulRectangular();
        await conGPU(t, async ({ device }) => {
            const A = Float32Array.from({ length: M * K }, (_, i) => ((i * 7) % 13) / 13 + 0.1);
            const B = Float32Array.from({ length: K * P }, (_, i) => ((i * 5) % 11) / 11 - 0.2);
            const esperado = productoCPU(A, B, M, P, K);
            const resultados: Record<string, Float32Array> = {};
            for (const [nombre, baldosas] of [['baldosas', true], ['ingenuo', false]] as const) {
                const wgsl = new EmisorWGSL({ baldosas }).emitir(kernel);
                const gpu = crearBuffers(device, kernel, { params: { M, P, K }, buffers: { A, B, C: M * P } });
                try {
                    await ejecutarKernel(device, kernel, wgsl, gpu, { M, P, K });
                    resultados[nombre] = (await leerSalidas(device, kernel, gpu))['C'] as Float32Array;
                } finally { gpu.destruir(); }
                cercanos(resultados[nombre]!, esperado, 1e-5, `${nombre} vs CPU (${M},${P},${K})`);
            }
            cercanos(resultados['baldosas']!, resultados['ingenuo']!, 1e-5, 'baldosas vs ingenuo');
        });
    });
}
