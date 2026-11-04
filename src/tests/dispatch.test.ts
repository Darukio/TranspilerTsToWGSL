import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { TranspiladorTsaWGSL } from '../core/TranspiladorTSaWGSL.js';
import { crearBuffers } from '../runtime/buffers.js';
import { calcularWorkgroups, ejecutarKernel, lanzarKernel, prepararKernel } from '../runtime/dispatch.js';
import { leerSalidas } from '../runtime/lectura.js';
import { describirDispositivo } from '../runtime/dispositivo.js';
import type { KernelIR } from '../ir/ir-contract.js';
import { EmisorWGSL } from '../backend/EmisorWGSL.js';
import { kernelNoCuadrado } from './ejemplos-ir.js';
import { conGPU, leer } from './ayudas-gpu.js';

const prueba = (nombre: string) => fileURLToPath(new URL(`../../src/pruebas/${nombre}.ts`, import.meta.url));

function transpilar(nombre: string) {
    const ctx = new TranspiladorTsaWGSL().transpilar([prueba(nombre)]);
    assert.deepEqual(ctx.diagnosticos, []);
    return { kernel: ctx.kernels[0]!, wgsl: ctx.shaders[0]!.wgsl };
}

/** Compara con tolerancia: la GPU trabaja en f32 y la CPU en f64. */
function cercanos(real: ArrayLike<number>, esperado: ArrayLike<number>, tolerancia: number) {
    assert.equal(real.length, esperado.length);
    for (let i = 0; i < esperado.length; i++) {
        const margen = tolerancia * Math.max(1, Math.abs(esperado[i]!));
        assert.ok(Math.abs(real[i]! - esperado[i]!) <= margen, `índice ${i}: GPU=${real[i]} CPU=${esperado[i]}`);
    }
}

test('calcularWorkgroups: ceil(tamaño / hilos) por eje', () => {
    const k1 = transpilar('vectores').kernel;
    assert.deepEqual(calcularWorkgroups(k1, { n: 64 }).grupos, [1, 1, 1]);
    assert.deepEqual(calcularWorkgroups(k1, { n: 65 }).grupos, [2, 1, 1]);
    assert.deepEqual(calcularWorkgroups(k1, { n: 1000 }).grupos, [16, 1, 1]);
    const k2 = transpilar('matrices').kernel;
    assert.deepEqual(calcularWorkgroups(k2, { N: 10 }), { grupos: [2, 2, 1], tamanoWorkgroup: [8, 8, 1] });
    assert.throws(() => calcularWorkgroups(k1, {}), /"n".*no tiene valor/);
});

test('vectores en la GPU coincide con la CPU (1000 elementos, 16 workgroups)', async (t) => {
    const { kernel, wgsl } = transpilar('vectores');
    await conGPU(t, async ({ device }) => {
        const n = 1000;
        const A = Float32Array.from({ length: n }, (_, i) => i * 0.5);
        const B = Float32Array.from({ length: n }, (_, i) => i % 7);
        const gpu = crearBuffers(device, kernel, { params: { n }, buffers: { vectorA: A, vectorB: B, salida: n } });
        try {
            const r = await ejecutarKernel(device, kernel, wgsl, gpu, { n });
            assert.deepEqual(r.grupos, [16, 1, 1]);
            const esperado = A.map((a, i) => a * 3.14 - B[i]!);
            cercanos(new Float32Array(await leer(device, gpu.buffers.get('salida')!, n * 4)), esperado, 1e-5);
        } finally { gpu.destruir(); }
    });
});

test('matrices en la GPU: grid 2D de 8x8 cubre N=10 sin desbordar', async (t) => {
    const { kernel, wgsl } = transpilar('matrices');
    await conGPU(t, async ({ device }) => {
        const N = 10;
        const A = Float32Array.from({ length: N * N }, (_, i) => i);
        const B = Float32Array.from({ length: N * N }, (_, i) => 1000 - i);
        const gpu = crearBuffers(device, kernel, { params: { N }, buffers: { A, B, C: N * N } });
        try {
            assert.deepEqual((await ejecutarKernel(device, kernel, wgsl, gpu, { N })).grupos, [2, 2, 1]);
            cercanos(new Float32Array(await leer(device, gpu.buffers.get('C')!, N * N * 4)), A.map((a, i) => a + B[i]!), 1e-6);
        } finally { gpu.destruir(); }
    });
});

test('monte-carlo en la GPU: sqrt y sin coinciden con Math.* dentro de tolerancia', async (t) => {
    const { kernel, wgsl } = transpilar('monte-carlo');
    await conGPU(t, async ({ device }) => {
        const n = 500;
        const precios = Float32Array.from({ length: n }, (_, i) => 1 + i * 0.01);
        const gpu = crearBuffers(device, kernel, { params: { n }, buffers: { precios, resultados: n } });
        try {
            await ejecutarKernel(device, kernel, wgsl, gpu, { n });
            const esperado = precios.map((p) => Math.sqrt(p * 2.5) + Math.sin(p));
            cercanos(new Float32Array(await leer(device, gpu.buffers.get('resultados')!, n * 4)), esperado, 1e-3);
        } finally { gpu.destruir(); }
    });
});

test('un shader inválido falla con el mensaje de compilación', async (t) => {
    const { kernel } = transpilar('vectores');
    await conGPU(t, async ({ device }) => {
        const gpu = crearBuffers(device, kernel, { params: { n: 4 }, buffers: { vectorA: [1, 2, 3, 4], vectorB: [1, 2, 3, 4], salida: 4 } });
        try {
            await assert.rejects(ejecutarKernel(device, kernel, '@compute fn roto( {', gpu, { n: 4 }), /El shader no compila/);
        } finally { gpu.destruir(); }
    });
});

test('leerSalidas devuelve solo los buffers que el kernel escribe, con su tipo', async (t) => {
    const { kernel, wgsl } = transpilar('vectores');
    await conGPU(t, async ({ device }) => {
        const gpu = crearBuffers(device, kernel, { params: { n: 4 }, buffers: { vectorA: [1, 2, 3, 4], vectorB: [1, 1, 1, 1], salida: 4 } });
        try {
            await ejecutarKernel(device, kernel, wgsl, gpu, { n: 4 });
            const salidas = await leerSalidas(device, kernel, gpu);
            assert.deepEqual(Object.keys(salidas), ['salida']);
            assert.ok(salidas['salida'] instanceof Float32Array);
            cercanos(salidas['salida']!, [1 * 3.14 - 1, 2 * 3.14 - 1, 3 * 3.14 - 1, 4 * 3.14 - 1], 1e-5);
        } finally { gpu.destruir(); }
    });
});

test('timestamp-query mide el cómputo en la GPU (si el dispositivo lo soporta)', async (t) => {
    const { kernel, wgsl } = transpilar('monte-carlo');
    await conGPU(t, async (dispositivo) => {
        const { device } = dispositivo;
        const n = 1 << 20;
        const precios = Float32Array.from({ length: n }, (_, i) => 1 + (i % 1000) * 0.01);
        const gpu = crearBuffers(device, kernel, { params: { n }, buffers: { precios, resultados: n } });
        try {
            const r = await ejecutarKernel(device, kernel, wgsl, gpu, { n });
            t.diagnostic(`${describirDispositivo(dispositivo)} | n=${n} | cómputo GPU=${r.msCompute?.toFixed(3)} ms | espera total=${r.msEspera.toFixed(3)} ms`);
            assert.ok(Number.isFinite(r.msEspera) && r.msEspera >= 0);
            if (dispositivo.timestampQuery) {
                assert.ok(r.msCompute !== undefined && Number.isFinite(r.msCompute) && r.msCompute >= 0, `msCompute=${r.msCompute}`);
            } else {
                assert.equal(r.msCompute, undefined);
            }
        } finally { gpu.destruir(); }
    });
});

test('lanzamientos repetidos sobre los mismos buffers dan el mismo resultado; k inválido falla', async (t) => {
    const { kernel, wgsl } = transpilar('vectores');
    await conGPU(t, async ({ device }) => {
        const gpu = crearBuffers(device, kernel, { params: { n: 4 }, buffers: { vectorA: [1, 2, 3, 4], vectorB: [1, 1, 1, 1], salida: 4 } });
        try {
            const preparado = await prepararKernel(device, kernel, wgsl);
            await lanzarKernel(device, preparado, gpu, { n: 4 }, { lanzamientos: 25 });
            cercanos((await leerSalidas(device, kernel, gpu))['salida']!, [2.14, 5.28, 8.42, 11.56], 1e-5);
            await assert.rejects(lanzarKernel(device, preparado, gpu, { n: 4 }, { lanzamientos: 0 }), /entero positivo/);
            await assert.rejects(lanzarKernel(device, preparado, gpu, { n: 4 }, { lanzamientos: 1.5 }), /entero positivo/);
        } finally { gpu.destruir(); }
    });
});

for (const filaMayor of [true, false]) {
    test(`grid no cuadrado 3×70 (${filaMayor ? 'fila mayor' : 'columna mayor'}): cada eje recibe su tamaño y el resultado es exacto`, async (t) => {
        const kernel = kernelNoCuadrado(filaMayor);
        const wgsl = new EmisorWGSL().emitir(kernel);
        const R = 3, C = 70;
        // x cubre el eje contiguo: 70 con filas mayores (2 workgroups de 8) o 3 con columnas mayores (1 workgroup)
        assert.deepEqual(calcularWorkgroups(kernel, { R, C }).grupos, filaMayor ? [9, 1, 1] : [1, 9, 1]);
        await conGPU(t, async ({ device }) => {
            const gpu = crearBuffers(device, kernel, { params: { R, C }, buffers: { out: R * C } });
            try {
                await ejecutarKernel(device, kernel, wgsl, gpu, { R, C });
                const real = (await leerSalidas(device, kernel, gpu))['out']!;
                const esperado = new Float32Array(R * C);
                for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) esperado[filaMayor ? i * C + j : j * R + i] = i * 1000 + j;
                assert.deepEqual([...real], [...esperado]);
            } finally { gpu.destruir(); }
        });
    });
}

for (const N of [1, 17, 64]) {
    test(`matmul real en la GPU (N=${N}) coincide con el producto de matrices en la CPU`, async (t) => {
        const { kernel, wgsl } = transpilar('matmul');
        await conGPU(t, async ({ device }) => {
            const A = Float32Array.from({ length: N * N }, (_, i) => ((i * 7) % 13) / 13);
            const B = Float32Array.from({ length: N * N }, (_, i) => ((i * 5) % 11) / 11);
            const esperado = new Float64Array(N * N);
            for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < N; k++) s += A[i * N + k]! * B[k * N + j]!; esperado[i * N + j] = s; }
            const gpu = crearBuffers(device, kernel, { params: { N }, buffers: { A, B, C: N * N } });
            try {
                await ejecutarKernel(device, kernel, wgsl, gpu, { N });
                cercanos((await leerSalidas(device, kernel, gpu))['C']!, esperado, 1e-5);
            } finally { gpu.destruir(); }
        });
    });
}
