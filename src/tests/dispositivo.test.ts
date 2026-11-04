import { test } from 'node:test';
import assert from 'node:assert/strict';
import v8 from 'node:v8';
import vm from 'node:vm';
import { crearDispositivo, describirDispositivo } from '../runtime/dispositivo.js';

test('crearDispositivo abre un dispositivo WebGPU usable (se omite sin GPU)', async (t) => {
    let dispositivo;
    try {
        dispositivo = await crearDispositivo();
    } catch (error) {
        if (error instanceof Error && error.message.startsWith('No hay adaptador WebGPU')) {
            t.skip(error.message);
            return;
        }
        throw error;
    }

    try {
        const modulo = dispositivo.device.createShaderModule({ code: '@compute @workgroup_size(1) fn main() {}' });
        const info = await modulo.getCompilationInfo();
        assert.equal(info.messages.filter(m => m.type === 'error').length, 0);
        t.diagnostic(describirDispositivo(dispositivo));
        assert.equal(typeof dispositivo.timestampQuery, 'boolean');
        assert.equal(dispositivo.device.features.has('timestamp-query'), dispositivo.timestampQuery);
    } finally {
        dispositivo.destruir();
    }
});

test('usa navigator.gpu cuando existe (camino del navegador)', async () => {
    const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
    let consultado = false;
    const gpuFalso = { requestAdapter: async () => { consultado = true; return null; } };
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: gpuFalso }, configurable: true });
    try {
        await assert.rejects(crearDispositivo(), /No hay adaptador WebGPU/);
        assert.equal(consultado, true);
    } finally {
        if (original) Object.defineProperty(globalThis, 'navigator', original);
    }
});

test('el dispositivo sigue vivo tras recolectar basura (regresión: Dawn se caía)', async (t) => {
    v8.setFlagsFromString('--expose-gc');
    const gc = vm.runInNewContext('gc') as () => void;

    let dispositivo;
    try {
        dispositivo = await crearDispositivo();
    } catch (error) {
        if (error instanceof Error && error.message.startsWith('No hay adaptador WebGPU')) return t.skip(error.message);
        throw error;
    }
    try {
        gc();
        await new Promise((resolve) => setTimeout(resolve, 50));
        gc();
        const buffer = dispositivo.device.createBuffer({ size: 16, usage: 0x08 | 0x04 });
        dispositivo.device.queue.writeBuffer(buffer, 0, new Float32Array([1, 2, 3, 4]));
        await dispositivo.device.queue.onSubmittedWorkDone();
        buffer.destroy();
    } finally {
        dispositivo.destruir();
    }
});

test('describirDispositivo marca los adaptadores por software', () => {
    const base = { timestampQuery: true, info: { vendor: 'mesa', architecture: 'software', device: 'llvmpipe', description: 'llvmpipe: Mesa' } };
    assert.match(describirDispositivo({ ...base, software: true }), /llvmpipe: Mesa \| SOFTWARE .*\| timestamp-query: sí/);
    assert.match(describirDispositivo({ ...base, software: false }), /\| hardware \|/);
});
