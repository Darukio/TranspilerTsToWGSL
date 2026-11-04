import { crearDispositivo, type Dispositivo } from '../runtime/dispositivo.js';

/** Ejecuta `prueba` con un dispositivo real; si no hay adaptador, omite el test. */
export async function conGPU(t: { skip(m?: string): void }, prueba: (d: Dispositivo) => Promise<void>) {
    let dispositivo: Dispositivo;
    try {
        dispositivo = await crearDispositivo();
    } catch (error) {
        if (error instanceof Error && error.message.startsWith('No hay adaptador WebGPU')) return t.skip(error.message);
        throw error;
    }
    try { await prueba(dispositivo); } finally { dispositivo.destruir(); }
}

export { leerBytes as leer } from '../runtime/lectura.js';
