export interface InfoAdaptador {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
}

export interface Dispositivo {
    device: GPUDevice;
    info: InfoAdaptador;
    /** true si el adaptador soporta medir tiempo de cómputo en la GPU. */
    timestampQuery: boolean;
    /** true si el "adaptador" es un emulador en CPU (llvmpipe, SwiftShader...): un benchmark contra la CPU no vale. */
    software: boolean;
    destruir(): void;
}

/**
 * En el navegador WebGPU ya viene en `navigator.gpu`. En Node no existe, así que se carga
 * Dawn (paquete `webgpu`) bajo demanda; si se empaqueta para web, marcar `webgpu` como externo.
 */
async function obtenerGPU(): Promise<GPU> {
    const navegador = (globalThis as { navigator?: { gpu?: GPU } }).navigator;
    if (navegador?.gpu) return navegador.gpu;
    const { create } = await import('webgpu');
    // Dawn redondea los timestamps a 65 µs (protección para páginas web). Acá corre solo en local y
    // medimos kernels de microsegundos, así que se desactiva para ver tiempos reales.
    return create(['disable-dawn-features=timestamp_quantization']);
}

const ES_SOFTWARE = /llvmpipe|swiftshader|software|lavapipe/i;

/** Una línea legible: qué GPU se detectó y si sirve para medir rendimiento. */
export function describirDispositivo(d: Pick<Dispositivo, 'info' | 'timestampQuery' | 'software'>): string {
    const { description, device, vendor } = d.info;
    const nombre = [description, device].filter((x) => x && !description.includes(x)).length
        ? `${description} - ${device}`
        : description || `${vendor} ${device}`.trim() || 'adaptador sin nombre';
    const tipo = d.software ? 'SOFTWARE (emulado en CPU, no sirve para comparar rendimiento)' : 'hardware';
    return `${nombre} | ${tipo} | timestamp-query: ${d.timestampQuery ? 'sí' : 'no'}`;
}

// Dawn libera el dispositivo si se recolectan el `gpu` y el `adapter` que lo crearon. Se retienen
// aquí hasta `destruir()`, aunque el llamador solo conserve `device`.
const retenidos = new Set<object>();

/**
 * Pide un adaptador WebGPU (navegador o Node) y abre un dispositivo.
 * Activa `timestamp-query` solo si el adaptador lo ofrece: es opcional.
 */
export async function crearDispositivo(): Promise<Dispositivo> {
    const gpu = await obtenerGPU();
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
        throw new Error('No hay adaptador WebGPU disponible (¿drivers Vulkan/Metal/D3D12 instalados?).');
    }

    const timestampQuery = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({
        requiredFeatures: timestampQuery ? ['timestamp-query'] : [],
    });

    const ref = { gpu, adapter };
    retenidos.add(ref);

    const { vendor, architecture, device: modelo, description } = adapter.info;
    const software = ES_SOFTWARE.test(`${architecture} ${modelo} ${description}`);
    return {
        device,
        info: { vendor, architecture, device: modelo, description },
        timestampQuery,
        software,
        destruir: () => {
            device.destroy();
            retenidos.delete(ref);
        },
    };
}
