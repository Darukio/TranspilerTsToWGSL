import { test } from 'node:test';
import assert from 'node:assert/strict';
import { EmisorWGSL } from '../backend/EmisorWGSL.js';
import type { KernelIR } from '../ir/ir-contract.js';

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
