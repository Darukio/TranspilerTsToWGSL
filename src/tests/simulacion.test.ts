import { test } from 'node:test';
import assert from 'node:assert/strict';
import { multiplicarMatricesReal } from '../pruebas/matmul.js';
import { difundirCalor } from '../pruebas/calor.js';
import { calcularFuerzas } from '../pruebas/nbody.js';
import { integrarCuerpos } from '../pruebas/integrar.js';
import { CALOR, NBODY, crearSimulacion, empaquetar, estampar, fuenteCalor, galaxia, imagenA, matrizB, parametrosB, placaInicial } from '../server/simulacion.js';
import { conGPU } from './ayudas-gpu.js';

const cuantizar = (v: ArrayLike<number>) => Uint8Array.from({ length: v.length }, (_, i) => Math.min(255, Math.max(0, Math.round(v[i]! * 255))));
/** Mayor diferencia entre dos imágenes de 8 bits (el redondeo de f32 contra f64 puede mover un nivel). */
const difMax = (a: Uint8Array, b: Uint8Array) => {
    assert.equal(a.length, b.length);
    return a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]!)), 0);
};

test('empaquetar: [largo JSON][largo payload][JSON][payload]', () => {
    const m = empaquetar({ meta: { tipo: 'x', n: 3 }, payload: Uint8Array.of(7, 8, 9) });
    const json = m.readUInt32LE(0), carga = m.readUInt32LE(4);
    assert.equal(carga, 3);
    assert.deepEqual(JSON.parse(m.subarray(8, 8 + json).toString()), { tipo: 'x', n: 3 });
    assert.deepEqual([...m.subarray(8 + json)], [7, 8, 9]);
});

test('matrizB: cada fila suma 1 y desplaza la imagen (A·B mueve las columnas)', () => {
    const N = 16;
    const B = matrizB(N, 3, 2);
    for (let k = 0; k < N; k++) assert.ok(Math.abs(B.slice(k * N, (k + 1) * N).reduce((a, b) => a + b, 0) - 1) < 1e-6);
    // ancho 0 = permutación pura: C[i][j] = A[i][j - 3]
    const P = matrizB(N, 3, 0);
    const A = imagenA(N), C = new Float64Array(N * N);
    multiplicarMatricesReal(Array.from(A), Array.from(P), C as unknown as number[], N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) assert.ok(Math.abs(C[i * N + j]! - A[i * N + ((j - 3 + N) % N)]!) < 1e-6);
});

test('la galaxia arranca con órbitas tangenciales alrededor del cuerpo central', () => {
    const { x, y, vx, vy } = galaxia(100);
    for (let i = 1; i < 100; i++) assert.ok(Math.abs(x[i]! * vx[i]! + y[i]! * vy[i]!) < 1e-4, 'la velocidad es perpendicular al radio');
});

test('simulación de matmul: el cuadro que dibuja es C = A·B calculado en la GPU', async (t) => {
    await conGPU(t, async (d) => {
        const N = 32;
        const sim = await crearSimulacion(d, 'matmul', N);
        try {
            assert.equal(sim.inicio.payload.length, N * N);
            const { payload, meta } = await sim.siguiente();
            const { desplazamiento, ancho } = parametrosB(N, 0);
            const C = new Float64Array(N * N);
            multiplicarMatricesReal(Array.from(imagenA(N)), Array.from(matrizB(N, desplazamiento, ancho)), C as unknown as number[], N);
            assert.equal(payload.length, 2 * N * N, 'C y B');
            assert.ok(difMax(payload.subarray(0, N * N), cuantizar(C)) <= 1);
            assert.equal(meta['trabajo'], 2 * N ** 3);
        } finally { sim.destruir(); }
    });
});

test('simulación de calor: un cuadro equivale a 8 pasos de la función original más la fuente', async (t) => {
    await conGPU(t, async (d) => {
        const N = 32;
        const sim = await crearSimulacion(d, 'calor', N);
        try {
            let actual = Array.from(placaInicial(N)), siguiente = Array.from(placaInicial(N));
            const { cx, cy } = fuenteCalor(N, 0);
            const campo = Float32Array.from(actual);
            estampar(campo, N, cx, cy, N * 0.03, 1);
            actual = Array.from(campo);
            for (let p = 0; p < CALOR.pasosPorCuadro; p++) { difundirCalor(actual, siguiente, N, CALOR.alfa); [actual, siguiente] = [siguiente, actual]; }
            const { payload } = await sim.siguiente();
            assert.ok(difMax(payload, cuantizar(actual)) <= 1);
            assert.ok(payload.some((v) => v > 0), 'hay calor en la placa');
        } finally { sim.destruir(); }
    });
});

test('simulación de n cuerpos: un cuadro equivale a los pasos de la CPU (fuerzas + integración)', async (t) => {
    await conGPU(t, async (d) => {
        const n = 64;
        const sim = await crearSimulacion(d, 'nbody', n);
        try {
            const { x, y, vx, vy, masa } = galaxia(n);
            const f = (a: Float32Array) => Float64Array.from(a) as unknown as number[];
            const [X, Y, VX, VY, M] = [f(x), f(y), f(vx), f(vy), f(masa)];
            const AX = new Float64Array(n) as unknown as number[], AY = new Float64Array(n) as unknown as number[];
            for (let p = 0; p < NBODY.pasosPorCuadro; p++) {
                calcularFuerzas(X, Y, M, AX, AY, n, NBODY.suavizado);
                integrarCuerpos(X, Y, VX, VY, AX, AY, n, NBODY.dt);
            }
            const { payload } = await sim.siguiente();
            const gpu = new Float32Array(payload.slice().buffer);
            assert.equal(gpu.length, 2 * n);
            for (let i = 0; i < n; i++) {
                assert.ok(Math.abs(gpu[i]! - X[i]!) < 2e-3, `x[${i}]: ${gpu[i]} contra ${X[i]}`);
                assert.ok(Math.abs(gpu[n + i]! - Y[i]!) < 2e-3, `y[${i}]: ${gpu[n + i]} contra ${Y[i]}`);
            }
        } finally { sim.destruir(); }
    });
});
