import { test } from 'node:test';
import assert from 'node:assert/strict';
import { crearBuffers } from '../runtime/buffers.js';
import { conGPU, leer } from './ayudas-gpu.js';
import { calcularBindings } from '../backend/Bindings.js';
import { EmisorWGSL } from '../backend/EmisorWGSL.js';
import type { KernelIR } from '../ir/ir-contract.js';

const kernel: KernelIR = {
    irVersion: 1,
    name: 'k',
    params: [{ name: 'n', type: 'i32' }, { name: 'escala', type: 'f32' }],
    buffers: [
        { name: 'a', type: 'f32', access: 'read' },
        { name: 'salida', type: 'f32', access: 'write' },
    ],
    grid: [{ iter: 'i', size: { op: 'ref', id: 'n' } }],
    body: [{ op: 'store', buffer: 'salida', index: { op: 'ref', id: 'i' }, value: { op: 'read', buffer: 'a', index: { op: 'ref', id: 'i' } } }],
};

test('los bindings del descriptor coinciden con los @binding del shader', () => {
    const bindings = calcularBindings(kernel);
    assert.equal(bindings.params?.binding, 0);
    assert.deepEqual(bindings.buffers.map((b) => [b.name, b.binding]), [['a', 1], ['salida', 2]]);
    const wgsl = new EmisorWGSL().emitir(kernel);
    assert.match(wgsl, /@binding\(0\) var<uniform> params/);
    assert.match(wgsl, /@binding\(1\) var<storage, read> a:/);
    assert.match(wgsl, /@binding\(2\) var<storage, read_write> salida:/);
});

test('sin parámetros el primer buffer usa el binding 0', () => {
    const sinParams: KernelIR = { ...kernel, params: [] };
    assert.deepEqual(calcularBindings(sinParams).buffers.map((b) => b.binding), [0, 1]);
});

test('valida la entrada antes de tocar la GPU', () => {
    const sinGPU = null as unknown as GPUDevice;
    assert.throws(() => crearBuffers(sinGPU, kernel, { params: { n: 2 }, buffers: {} }), /Falta el parámetro "escala"/);
    assert.throws(() => crearBuffers(sinGPU, kernel, { params: { n: 2, escala: 1 }, buffers: { a: [1, 2] } }), /Faltan los datos del buffer "salida"/);
    assert.throws(() => crearBuffers(sinGPU, kernel, { params: { n: 2, escala: 1 }, buffers: { a: [], salida: [0] } }), /"a" está vacío/);
    assert.throws(() => crearBuffers(sinGPU, kernel, { params: { n: 2, escala: 1 }, buffers: { a: 4, salida: 4 } }), /"a" es de solo lectura/);
    assert.throws(() => crearBuffers(sinGPU, kernel, { params: { n: 2, escala: 1 }, buffers: { a: [1], salida: 0 } }), /largo del buffer "salida"/);
});

test('sube params y buffers a la GPU con los valores correctos', async (t) => {
    await conGPU(t, async ({ device }) => {
        const gpu = crearBuffers(device, kernel, { params: { n: 3, escala: 2.5 }, buffers: { a: [1, 2, 3], salida: [0, 0, 0] } });
        try {
            const params = new DataView(await leer(device, gpu.params!, 16));
            assert.equal(params.getInt32(0, true), 3);
            assert.equal(params.getFloat32(4, true), 2.5);
            assert.deepEqual([...new Float32Array(await leer(device, gpu.buffers.get('a')!, 12))], [1, 2, 3]);
            assert.deepEqual([...new Float32Array(await leer(device, gpu.buffers.get('salida')!, 12))], [0, 0, 0]);
        } finally {
            gpu.destruir();
        }
    });
});

test('un buffer de salida se crea solo con su largo y nace en cero', async (t) => {
    await conGPU(t, async ({ device }) => {
        const gpu = crearBuffers(device, kernel, { params: { n: 3, escala: 1 }, buffers: { a: [1, 2, 3], salida: 5 } });
        try {
            assert.equal(gpu.buffers.get('salida')!.size, 20);
            assert.deepEqual([...new Float32Array(await leer(device, gpu.buffers.get('salida')!, 20))], [0, 0, 0, 0, 0]);
        } finally {
            gpu.destruir();
        }
    });
});
