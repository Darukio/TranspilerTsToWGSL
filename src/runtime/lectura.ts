import type { DataType, KernelIR } from '../ir/ir-contract.js';
import { calcularBindings } from '../backend/Bindings.js';
import { USO, type BuffersGPU } from './buffers.js';

export type Tipado = Float32Array | Int32Array | Uint32Array;

const TIPADOS = { f32: Float32Array, i32: Int32Array, u32: Uint32Array } as const;

/** Copia un buffer de la GPU a la CPU. Hace falta un buffer intermedio MAP_READ: los de storage no se pueden mapear. */
export async function leerBytes(device: GPUDevice, origen: GPUBuffer, bytes: number = origen.size): Promise<ArrayBuffer> {
    const destino = device.createBuffer({ size: bytes, usage: USO.MAP_READ | USO.COPY_DST, label: 'lectura' });
    try {
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(origen, 0, destino, 0, bytes);
        device.queue.submit([encoder.finish()]);
        await destino.mapAsync(1); // GPUMapMode.READ
        return destino.getMappedRange().slice(0);
    } finally {
        destino.destroy();
    }
}

export async function leerBuffer(device: GPUDevice, origen: GPUBuffer, tipo: DataType): Promise<Tipado> {
    return new TIPADOS[tipo](await leerBytes(device, origen));
}

/** Lee los buffers que el kernel escribe (`write` / `readwrite`), con el tipo de elemento de cada uno. */
export async function leerSalidas(device: GPUDevice, kernel: KernelIR, gpu: BuffersGPU): Promise<Record<string, Tipado>> {
    const salidas: Record<string, Tipado> = {};
    for (const b of calcularBindings(kernel).buffers) {
        if (b.access === 'read') continue;
        const buffer = gpu.buffers.get(b.name);
        if (!buffer) throw new Error(`No hay buffer en la GPU para "${b.name}"`);
        salidas[b.name] = await leerBuffer(device, buffer, b.type);
    }
    return salidas;
}
