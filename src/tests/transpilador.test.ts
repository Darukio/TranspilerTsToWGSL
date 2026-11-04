import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { EmisorWGSL } from '../backend/EmisorWGSL.js';
import { TranspiladorTsaWGSL } from '../core/TranspiladorTSaWGSL.js';

const prueba = (nombre: string) => fileURLToPath(new URL(`../../src/pruebas/${nombre}.ts`, import.meta.url));

function transpilar(nombre: string) {
    const ctx = new TranspiladorTsaWGSL().transpilar([prueba(nombre)]);
    assert.deepEqual(ctx.diagnosticos, []);
    assert.equal(ctx.shaders.length, 1);
    return ctx.shaders[0]!.wgsl;
}

test('vectores: un hilo por elemento, buffers read/read_write y literal f32', () => {
    const wgsl = transpilar('vectores');
    assert.match(wgsl, /struct Params \{\n  n: i32,\n\};/);
    assert.match(wgsl, /var<storage, read> vectorA: array<f32>;/);
    assert.match(wgsl, /var<storage, read> vectorB: array<f32>;/);
    assert.match(wgsl, /var<storage, read_write> salida: array<f32>;/);
    assert.match(wgsl, /let i = i32\(gid\.x\);/);
    assert.match(wgsl, /if \(i >= params\.n\) \{ return; \}/);
    assert.match(wgsl, /salida\[i\] = \(\(vectorA\[i\] \* 3\.14\) - vectorB\[i\]\);/);
});

test('matrices: bucles anidados forman un grid 2D', () => {
    const wgsl = transpilar('matrices');
    // x recorre j (el índice contiguo en C[i * N + j]); y recorre i
    assert.match(wgsl, /let j = i32\(gid\.x\);/);
    assert.match(wgsl, /let i = i32\(gid\.y\);/);
    assert.match(wgsl, /if \(i >= params\.N \|\| j >= params\.N\) \{ return; \}/);
    assert.match(wgsl, /C\[\(\(i \* params\.N\) \+ j\)\] = /);
});

test('monte-carlo: Math.* se traduce a funciones WGSL y 2.5 queda f32', () => {
    const wgsl = transpilar('monte-carlo');
    assert.match(wgsl, /resultados\[i\] = \(sqrt\(\(precios\[i\] \* 2\.5\)\) \+ sin\(precios\[i\]\)\);/);
});

test('un bucle con dependencia entre iteraciones se descarta con diagnóstico', () => {
    const ctx = new TranspiladorTsaWGSL().transpilar([fileURLToPath(new URL('../../src/tests/fixtures/dependiente.ts', import.meta.url))]);
    assert.equal(ctx.shaders.length, 0);
    assert.match(ctx.diagnosticos.map(d => d.mensaje).join('\n'), /dependencia entre iteraciones/);
});

test('matmul (kernel ingenuo): acumulador local e inner loop se traducen a var/for/set en WGSL', () => {
    const ctx = new TranspiladorTsaWGSL().transpilar([prueba('matmul')]);
    const wgsl = new EmisorWGSL({ baldosas: false }).emitir(ctx.kernels[0]!);
    assert.match(wgsl, /let j = i32\(gid\.x\);/); // x recorre j: B[k * N + j] y C[i * N + j] son contiguos en j
    assert.match(wgsl, /let i = i32\(gid\.y\);/);
    assert.match(wgsl, /var suma: f32 = 0\.0;/);
    assert.match(wgsl, /for \(var k = 0; k < params\.N; k\+\+\) \{/);
    assert.match(wgsl, /suma = \(suma \+ \(A\[\(\(i \* params\.N\) \+ k\)\] \* B\[\(\(k \* params\.N\) \+ j\)\]\)\);/);
    assert.match(wgsl, /C\[\(\(i \* params\.N\) \+ j\)\] = suma;/);
});

test('matmul: por defecto el transpilador emite el kernel con baldosas en memoria compartida y lo informa', () => {
    const ctx = new TranspiladorTsaWGSL().transpilar([prueba('matmul')]);
    assert.deepEqual(ctx.diagnosticos, []);
    assert.match(ctx.shaders[0]!.wgsl, /var<workgroup> tileA_/);
    assert.deepEqual(ctx.shaders[0]!.optimizaciones, ['baldosas 16×16 en memoria compartida']);
    // el resto de los casos no cambia
    assert.deepEqual(new TranspiladorTsaWGSL().transpilar([prueba('vectores')]).shaders[0]!.optimizaciones, []);
});

test('una reducción sobre una variable declarada fuera del bucle se descarta (sería una carrera entre hilos)', () => {
    const ctx = new TranspiladorTsaWGSL().transpilar([fileURLToPath(new URL('../../src/tests/fixtures/reduccion.ts', import.meta.url))]);
    assert.equal(ctx.shaders.length, 0);
    assert.match(ctx.diagnosticos.map((d) => d.mensaje).join('\n'), /«total».*fuera del bucle/);
});
