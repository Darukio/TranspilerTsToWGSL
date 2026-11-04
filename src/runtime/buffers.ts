import type { KernelIR, DataType } from '../ir/ir-contract.js';
import { calcularBindings } from '../backend/Bindings.js';

// Valores de GPUBufferUsage fijados por la especificación. Se usan como números para no depender
// de que el global exista (en Node no existe salvo que se registre).
export const USO = { MAP_READ: 0x01, COPY_SRC: 0x04, COPY_DST: 0x08, UNIFORM: 0x40, STORAGE: 0x80, QUERY_RESOLVE: 0x200 } as const;

export interface EntradaKernel {
    /** Valores de los parámetros escalares del kernel (por nombre). */
    params?: Record<string, number>;
    /**
     * Contenido de cada buffer (por nombre). Un buffer que el kernel escribe (`write` o `readwrite`)
     * puede indicarse solo con su largo en elementos: nace en cero, sin subir nada.
     */
    buffers: Record<string, ArrayLike<number> | number>;
}

export interface BuffersGPU {
    params?: GPUBuffer;
    buffers: Map<string, GPUBuffer>;
    destruir(): void;
}

const TIPADOS = { f32: Float32Array, i32: Int32Array, u32: Uint32Array } as const;

function aTipado(tipo: DataType, datos: ArrayLike<number>) {
    const Ctor = TIPADOS[tipo];
    return datos instanceof Ctor ? datos : Ctor.from(datos);
}

/** Empaqueta los parámetros como el `struct Params` de WGSL: un escalar de 4 bytes por campo, en orden. */
function empaquetarParams(campos: { name: string; type: DataType }[], valores: Record<string, number>): ArrayBuffer {
    const bytes = new ArrayBuffer(Math.ceil((campos.length * 4) / 16) * 16);
    const vista = new DataView(bytes);
    campos.forEach((c, i) => {
        const valor = valores[c.name];
        if (valor === undefined) throw new Error(`Falta el parámetro "${c.name}"`);
        if (c.type === 'f32') vista.setFloat32(i * 4, valor, true);
        else if (c.type === 'i32') vista.setInt32(i * 4, valor, true);
        else vista.setUint32(i * 4, valor, true);
    });
    return bytes;
}

/**
 * Crea y sube los buffers que el kernel declara, con los mismos `@binding` que el shader.
 * Valida la entrada completa antes de pedirle nada a la GPU.
 */
export function crearBuffers(device: GPUDevice, kernel: KernelIR, entrada: EntradaKernel): BuffersGPU {
    const bindings = calcularBindings(kernel);

    const paramsBytes = bindings.params ? empaquetarParams(bindings.params.campos, entrada.params ?? {}) : undefined;
    const datos = bindings.buffers.map((b) => {
        const crudo = entrada.buffers[b.name];
        if (crudo === undefined) throw new Error(`Faltan los datos del buffer "${b.name}"`);
        if (typeof crudo === 'number') {
            if (b.access === 'read') throw new Error(`El buffer "${b.name}" es de solo lectura: hay que pasar sus datos, no solo el largo`);
            if (!Number.isInteger(crudo) || crudo <= 0) throw new Error(`El largo del buffer "${b.name}" debe ser un entero positivo`);
            return { nombre: b.name, bytes: crudo * 4, tipado: undefined };
        }
        if (crudo.length === 0) throw new Error(`El buffer "${b.name}" está vacío`);
        const tipado = aTipado(b.type, crudo);
        return { nombre: b.name, bytes: tipado.byteLength, tipado };
    });

    const creados: GPUBuffer[] = [];
    const crear = (size: number, usage: number, label: string): GPUBuffer => {
        const buffer = device.createBuffer({ size, usage, label });
        creados.push(buffer);
        return buffer;
    };

    const resultado: BuffersGPU = {
        buffers: new Map(),
        destruir: () => creados.forEach((b) => b.destroy()),
    };

    if (paramsBytes) {
        resultado.params = crear(paramsBytes.byteLength, USO.UNIFORM | USO.COPY_DST | USO.COPY_SRC, 'params');
        device.queue.writeBuffer(resultado.params, 0, paramsBytes);
    }
    for (const { nombre, bytes, tipado } of datos) {
        const buffer = crear(bytes, USO.STORAGE | USO.COPY_DST | USO.COPY_SRC, nombre);
        // Un buffer nuevo ya está en cero: solo se sube si hay datos.
        if (tipado) device.queue.writeBuffer(buffer, 0, tipado as GPUAllowSharedBufferSource);
        resultado.buffers.set(nombre, buffer);
    }
    return resultado;
}
