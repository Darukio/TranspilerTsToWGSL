import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { TranspiladorTsaWGSL } from '../core/TranspiladorTSaWGSL.js';
import { crearBuffers } from '../runtime/buffers.js';
import { bufferFinal, ejecutarKernel, lanzarKernel, prepararKernel } from '../runtime/dispatch.js';
import { leerBuffer, leerSalidas } from '../runtime/lectura.js';
import { difundirCalor } from '../pruebas/calor.js';
import { calcularFuerzas } from '../pruebas/nbody.js';
import { conGPU } from './ayudas-gpu.js';

const transpilar = (archivo: string) => {
    const ctx = new TranspiladorTsaWGSL().transpilar([fileURLToPath(new URL(`../../src/pruebas/${archivo}.ts`, import.meta.url))]);
    assert.deepEqual(ctx.diagnosticos, []);
    return { kernel: ctx.kernels[0]!, wgsl: ctx.shaders[0]!.wgsl };
};
const errorRelativoMax = (real: ArrayLike<number>, esperado: ArrayLike<number>) => {
    assert.equal(real.length, esperado.length);
    let m = 0;
    for (let i = 0; i < esperado.length; i++) m = Math.max(m, Math.abs(real[i]! - esperado[i]!) / Math.max(1, Math.abs(esperado[i]!)));
    return m;
};

// ---------- P1: n-cuerpos ----------

for (const n of [1, 63, 300]) {
    test(`n-cuerpos en la GPU (n=${n}) coincide con la CPU`, async (t) => {
        const { kernel, wgsl } = transpilar('nbody');
        await conGPU(t, async ({ device }) => {
            const x = Float32Array.from({ length: n }, (_, i) => ((i * 7919) % 10007) / 100);
            const y = Float32Array.from({ length: n }, (_, i) => ((i * 104729) % 10009) / 100);
            const masa = Float32Array.from({ length: n }, (_, i) => 1 + (i % 5));
            const suavizado = 0.01;
            // referencia: la función original en f64 sobre los mismos datos
            const ax = new Float64Array(n), ay = new Float64Array(n);
            calcularFuerzas(Array.from(x), Array.from(y), Array.from(masa), ax as unknown as number[], ay as unknown as number[], n, suavizado);
            const gpu = crearBuffers(device, kernel, { params: { n, suavizado }, buffers: { x, y, masa, ax: n, ay: n } });
            try {
                await ejecutarKernel(device, kernel, wgsl, gpu, { n, suavizado });
                const salidas = await leerSalidas(device, kernel, gpu);
                // la suma de n términos de signos mezclados en f32 pierde dígitos: se acepta 1e-3 relativo al máximo
                assert.ok(errorRelativoMax(salidas['ax']!, ax) < 1e-3, `ax: ${errorRelativoMax(salidas['ax']!, ax)}`);
                assert.ok(errorRelativoMax(salidas['ay']!, ay) < 1e-3, `ay: ${errorRelativoMax(salidas['ay']!, ay)}`);
            } finally { gpu.destruir(); }
        });
    });
}

// ---------- P2: difusión de calor con doble búfer ----------

function placa(N: number): Float32Array {
    // calor en el centro y un borde frío fijo; los dos buffers arrancan iguales para que los bordes coincidan
    return Float32Array.from({ length: N * N }, (_, k) => {
        const i = Math.floor(k / N), j = k % N;
        return i === 0 || j === 0 || i === N - 1 || j === N - 1 ? 0 : ((i * 31 + j * 17) % 100) / 100;
    });
}

for (const pasos of [1, 2, 7, 8]) {
    test(`calor: ${pasos} paso(s) con doble búfer en la GPU coinciden con la simulación en la CPU`, async (t) => {
        const { kernel, wgsl } = transpilar('calor');
        await conGPU(t, async ({ device }) => {
            const N = 40, alfa = 0.1;
            const inicial = placa(N);
            // CPU: el mismo esquema de doble búfer con la función original
            let actual = Array.from(inicial), siguiente = Array.from(inicial);
            for (let p = 0; p < pasos; p++) { difundirCalor(actual, siguiente, N, alfa); [actual, siguiente] = [siguiente, actual]; }
            const gpu = crearBuffers(device, kernel, { params: { N, alfa }, buffers: { actual: inicial, siguiente: inicial } });
            try {
                const preparado = await prepararKernel(device, kernel, wgsl);
                const par: [string, string] = ['actual', 'siguiente'];
                await lanzarKernel(device, preparado, gpu, { N, alfa }, { lanzamientos: pasos, intercambio: par });
                const final = await leerBuffer(device, gpu.buffers.get(bufferFinal(par, pasos))!, 'f32');
                assert.ok(errorRelativoMax(final, actual) < 1e-5, `error ${errorRelativoMax(final, actual)}`);
                // la simulación avanzó de verdad: el campo ya no es el inicial
                assert.ok(errorRelativoMax(final, inicial) > 1e-3);
            } finally { gpu.destruir(); }
        });
    });
}

test('intercambio: valida los nombres y que los buffers tengan el mismo tamaño', async (t) => {
    const { kernel, wgsl } = transpilar('calor');
    await conGPU(t, async ({ device }) => {
        const gpu = crearBuffers(device, kernel, { params: { N: 8, alfa: 0.1 }, buffers: { actual: placa(8), siguiente: placa(8) } });
        try {
            const preparado = await prepararKernel(device, kernel, wgsl);
            await assert.rejects(lanzarKernel(device, preparado, gpu, { N: 8, alfa: 0.1 }, { intercambio: ['actual', 'nada'] }), /no tiene/);
        } finally { gpu.destruir(); }
        const desiguales = crearBuffers(device, kernel, { params: { N: 8, alfa: 0.1 }, buffers: { actual: placa(8), siguiente: 10 } });
        try {
            const preparado = await prepararKernel(device, kernel, wgsl);
            await assert.rejects(lanzarKernel(device, preparado, desiguales, { N: 8, alfa: 0.1 }, { intercambio: ['actual', 'siguiente'] }), /mismo tamaño/);
        } finally { desiguales.destruir(); }
    });
});

test('bufferFinal: con pasos impares el estado está en el segundo buffer, con pares en el primero', () => {
    assert.equal(bufferFinal(['a', 'b'], 1), 'b');
    assert.equal(bufferFinal(['a', 'b'], 2), 'a');
    assert.equal(bufferFinal(['a', 'b'], 7), 'b');
});
