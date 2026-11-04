import { test } from 'node:test';
import assert from 'node:assert/strict';
import { EmisorWGSL } from '../backend/EmisorWGSL.js';
import type { KernelIR } from '../ir/ir-contract.js';
import { asignarEjes } from '../backend/Bindings.js';
import { kernelNoCuadrado } from './ejemplos-ir.js';
import { TranspiladorTsaWGSL } from '../core/TranspiladorTSaWGSL.js';
import { fileURLToPath } from 'node:url';
import type { Expr } from '../ir/ir-contract.js';

const lit = (value: number) => ({ op: 'lit', type: 'i32', value }) as const;

function kernel(ejes: string[], buffer = 'out'): KernelIR {
    return {
        irVersion: 1,
        name: 'k',
        params: [],
        buffers: [{ name: buffer, type: 'f32', access: 'write' }],
        grid: ejes.map(iter => ({ iter, size: lit(4) })),
        body: [{ op: 'store', buffer, index: { op: 'ref', id: ejes[0]! }, value: { op: 'lit', type: 'f32', value: 1 } }],
    };
}

test('workgroup_size depende de las dimensiones del grid', () => {
    assert.match(new EmisorWGSL().emitir(kernel(['i'])), /@workgroup_size\(64, 1, 1\)/);
    assert.match(new EmisorWGSL().emitir(kernel(['i', 'j'])), /@workgroup_size\(8, 8, 1\)/);
    assert.match(new EmisorWGSL().emitir(kernel(['i', 'j', 'k'])), /@workgroup_size\(4, 4, 4\)/);
});

test('rechaza identificadores reservados de WGSL', () => {
    assert.throws(() => new EmisorWGSL().emitir(kernel(['fn'])), /reservado.*"fn"/);
    assert.throws(() => new EmisorWGSL().emitir(kernel(['i'], 'params')), /reservado.*"params"/);
});

test('rechaza escritura sobre un buffer de solo lectura', () => {
    const k = kernel(['i']);
    k.buffers[0]!.access = 'read';
    assert.throws(() => new EmisorWGSL().emitir(k), /solo lectura/);
});

test('asignarEjes: x va al iterador contiguo en memoria', () => {
    assert.deepEqual(asignarEjes(kernelNoCuadrado(true)), [1, 0]); // i * C + j  → x = j
    assert.deepEqual(asignarEjes(kernelNoCuadrado(false)), [0, 1]); // j * R + i → x = i
    assert.deepEqual(asignarEjes(kernel(['i'])), [0]);
    // sin información de paso (a[i] y b[j] por igual) se conserva el orden original
    const empate = kernel(['i', 'j']);
    empate.body.push({ op: 'store', buffer: 'out', index: { op: 'ref', id: 'j' }, value: { op: 'lit', type: 'f32', value: 1 } });
    assert.deepEqual(asignarEjes(empate), [0, 1]);
});

test('el emisor declara cada iterador con el eje que asignarEjes eligió', () => {
    const wgsl = new EmisorWGSL().emitir(kernelNoCuadrado(true));
    assert.match(wgsl, /let j = i32\(gid\.x\);\n\s+let i = i32\(gid\.y\);/);
    assert.match(new EmisorWGSL().emitir(kernelNoCuadrado(false)), /let i = i32\(gid\.x\);\n\s+let j = i32\(gid\.y\);/);
});

test('dos bucles hermanos pueden declarar la misma variable local (cada for tiene su ámbito)', () => {
    const entero = (value: number) => ({ op: 'lit', type: 'i32', value }) as const;
    const bucle = (iter: string): KernelIR['body'][number] => ({
        op: 'for', iter, from: entero(0), to: entero(3),
        body: [{ op: 'var', id: 't', type: 'f32', init: { op: 'lit', type: 'f32', value: 1 } }, { op: 'set', id: 't', value: { op: 'lit', type: 'f32', value: 2 } }],
    });
    const k = kernel(['i']);
    k.body = [bucle('a'), bucle('b')];
    const wgsl = new EmisorWGSL().emitir(k);
    assert.equal(wgsl.match(/var t: f32 = 1\.0;/g)?.length, 2);
});

test('asignarEjes sigue las variables locales: x va al iterador contiguo aunque el índice pase por una local', () => {
    const ref = (id: string): Expr => ({ op: 'ref', id });
    const k = kernelNoCuadrado(true); // out[i*C + j]: j contiguo → x = j
    // mismo índice, pero guardado en `c` (como en: const c = i*C + j; out[c] = …)
    const indice = (k.body[0] as { index: Expr }).index;
    k.body = [
        { op: 'var', id: 'c', type: 'i32', init: indice },
        { op: 'store', buffer: 'out', index: ref('c'), value: { op: 'lit', type: 'f32', value: 1 } },
    ];
    assert.deepEqual(asignarEjes(k), [1, 0]);
    // si la local se reasigna, no se sabe qué vale: se conserva el orden original
    k.body.push({ op: 'set', id: 'c', value: ref('c') });
    assert.deepEqual(asignarEjes(k), [0, 1]);
});

test('calor: el kernel del stencil recorre las columnas con gid.x (acceso contiguo)', () => {
    const ctx = new TranspiladorTsaWGSL().transpilar([fileURLToPath(new URL('../../src/pruebas/calor.ts', import.meta.url))]);
    assert.deepEqual(ctx.diagnosticos, []);
    assert.match(ctx.shaders[0]!.wgsl, /let j = i32\(gid\.x\);/);
    assert.match(ctx.shaders[0]!.wgsl, /let i = i32\(gid\.y\);/);
});
