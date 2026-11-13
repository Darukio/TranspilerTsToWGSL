import { fileURLToPath } from 'node:url';
import type { KernelIR } from '../ir/ir-contract.js';
import { TranspiladorTsaWGSL } from '../core/TranspiladorTSaWGSL.js';
import { crearBuffers, type BuffersGPU } from '../runtime/buffers.js';
import { bufferFinal, lanzarKernel, prepararKernel, type KernelPreparado } from '../runtime/dispatch.js';
import { leerBytes } from '../runtime/lectura.js';
import type { Dispositivo } from '../runtime/dispositivo.js';

/** Un mensaje del stream: encabezado o cuadro. `payload` son bytes crudos cuyo formato depende del caso. */
export interface Mensaje {
    meta: Record<string, unknown>;
    payload: Uint8Array;
}

export interface Simulacion {
    /** Primer mensaje: parámetros de la escena (y, en matmul, la matriz A). */
    inicio: Mensaje;
    /** Avanza la simulación en la GPU y devuelve el estado para dibujar. */
    siguiente(): Promise<Mensaje>;
    destruir(): void;
}

export interface InfoSimulacion {
    id: 'nbody' | 'calor' | 'matmul';
    titulo: string;
    tamanos: number[];
    defecto: number;
}

export const SIMULACIONES: InfoSimulacion[] = [
    { id: 'nbody', titulo: 'N cuerpos (gravedad 2D)', tamanos: [512, 1_024, 2_048, 4_096, 8_192], defecto: 2_048 },
    { id: 'calor', titulo: 'Difusión de calor', tamanos: [128, 256, 512], defecto: 256 },
    { id: 'matmul', titulo: 'Multiplicación de matrices', tamanos: [128, 256, 512, 1_024], defecto: 256 },
];

/** Formato de un mensaje en el cable: [u32 largo del JSON][u32 largo del payload][JSON][payload], enteros little-endian. */
export function empaquetar({ meta, payload }: Mensaje): Buffer {
    const json = Buffer.from(JSON.stringify(meta));
    const salida = Buffer.alloc(8 + json.length + payload.length);
    salida.writeUInt32LE(json.length, 0);
    salida.writeUInt32LE(payload.length, 4);
    json.copy(salida, 8);
    salida.set(payload, 8 + json.length);
    return salida;
}

// ---------- kernels ----------

const cache = new Map<string, { kernel: KernelIR; wgsl: string }>();

/** Transpila un archivo de `src/pruebas` (≈1 s por el `ts.Program`) y recuerda el resultado. */
function transpilar(archivo: string): { kernel: KernelIR; wgsl: string } {
    const previo = cache.get(archivo);
    if (previo) return previo;
    const ruta = fileURLToPath(new URL(`../../src/pruebas/${archivo}.ts`, import.meta.url));
    const ctx = new TranspiladorTsaWGSL().transpilar([ruta]);
    const kernel = ctx.kernels[0];
    const shader = ctx.shaders[0];
    if (!kernel || !shader) throw new Error(`No se pudo transpilar "${archivo}": ${JSON.stringify(ctx.diagnosticos)}`);
    const nuevo = { kernel, wgsl: shader.wgsl };
    cache.set(archivo, nuevo);
    return nuevo;
}

interface Kernel {
    preparado: KernelPreparado;
    kernel: KernelIR;
}
async function preparar(dispositivo: Dispositivo, archivo: string): Promise<Kernel> {
    const { kernel, wgsl } = transpilar(archivo);
    return { kernel, preparado: await prepararKernel(dispositivo.device, kernel, wgsl) };
}

/** Generador pseudoaleatorio determinista (mulberry32): la misma escena cada vez, sin depender de `Math.random`. */
function aleatorio(semilla: number): () => number {
    let s = semilla >>> 0;
    return () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// ---------- n cuerpos ----------

export const NBODY = { dt: 0.004, suavizado: 0.01, pasosPorCuadro: 5, radioVista: 4.5 } as const;

/** Un disco de cuerpos livianos en órbita circular alrededor de una masa central (el cuerpo 0). */
export function galaxia(n: number) {
    const rnd = aleatorio(12345);
    const masaCentral = 1;
    const masaDisco = 0.5;
    const masa = new Float32Array(n).fill(masaDisco / n);
    masa[0] = masaCentral;
    const radios = Array.from({ length: Math.max(0, n - 1) }, () => 0.4 + 3.6 * Math.sqrt(rnd())).sort((a, b) => a - b);
    const x = new Float32Array(n), y = new Float32Array(n), vx = new Float32Array(n), vy = new Float32Array(n);
    radios.forEach((r, k) => {
        const i = k + 1;
        const ang = rnd() * 2 * Math.PI;
        // velocidad circular con la masa que hay adentro de la órbita (centro + cuerpos más internos)
        const v = Math.sqrt((masaCentral + masaDisco * (k / n)) / r);
        x[i] = r * Math.cos(ang); y[i] = r * Math.sin(ang);
        vx[i] = -v * Math.sin(ang); vy[i] = v * Math.cos(ang);
    });
    return { x, y, vx, vy, masa };
}

async function simularNBody(dispositivo: Dispositivo, n: number): Promise<Simulacion> {
    const { device } = dispositivo;
    const fuerzas = await preparar(dispositivo, 'nbody');
    const integrar = await preparar(dispositivo, 'integrar');
    const { x, y, vx, vy, masa } = galaxia(n);
    const { dt, suavizado, pasosPorCuadro } = NBODY;
    const gpuF = crearBuffers(device, fuerzas.kernel, { params: { n, suavizado }, buffers: { x, y, masa, ax: n, ay: n } });
    let gpuI: BuffersGPU | undefined;
    try {
        // Los dos kernels comparten los mismos buffers de la GPU: las posiciones nunca vuelven a la CPU salvo para dibujar.
        gpuI = crearBuffers(device, integrar.kernel, { params: { n, dt }, buffers: { x, y, vx, vy, ax: new Float32Array(n), ay: new Float32Array(n) } });
    } catch (e) {
        gpuF.destruir();
        throw e;
    }
    // integrar usa sus propios buffers de x/y/ax/ay: se reemplazan por los de fuerzas para compartir el estado
    for (const nombre of ['x', 'y', 'ax', 'ay']) {
        gpuI.buffers.get(nombre)!.destroy();
        gpuI.buffers.set(nombre, gpuF.buffers.get(nombre)!);
    }
    const gpuIntegrar = gpuI;

    return {
        inicio: { meta: { tipo: 'nbody', n, radioVista: NBODY.radioVista, pasosPorCuadro }, payload: new Uint8Array(0) },
        async siguiente() {
            let ms = 0;
            for (let p = 0; p < pasosPorCuadro; p++) {
                ms += (await lanzarKernel(device, fuerzas.preparado, gpuF, { n, suavizado }, { timestamps: false })).msEspera;
                ms += (await lanzarKernel(device, integrar.preparado, gpuIntegrar, { n, dt }, { timestamps: false })).msEspera;
            }
            const inicioLectura = performance.now();
            const [bx, by] = await Promise.all([leerBytes(device, gpuF.buffers.get('x')!), leerBytes(device, gpuF.buffers.get('y')!)]);
            const payload = new Uint8Array(bx.byteLength + by.byteLength);
            payload.set(new Uint8Array(bx), 0);
            payload.set(new Uint8Array(by), bx.byteLength);
            return { meta: { tipo: 'cuadro', ms, msLectura: performance.now() - inicioLectura, trabajo: pasosPorCuadro * n * n, unidad: 'interacciones' }, payload };
        },
        destruir() {
            gpuF.destruir();
            gpuIntegrar.destruir();
        },
    };
}

// ---------- difusión de calor ----------

export const CALOR = { alfa: 0.1, /** par: así el estado siempre queda en `actual`. */ pasosPorCuadro: 8 } as const;

/** Placa fría con la misma temperatura inicial en los dos buffers (los bordes quedan fijos en 0). */
export function placaInicial(N: number): Float32Array {
    const placa = new Float32Array(N * N);
    const rnd = aleatorio(777);
    for (let b = 0; b < 4; b++) estampar(placa, N, 0.15 * N + rnd() * 0.7 * N, 0.15 * N + rnd() * 0.7 * N, N * 0.05, 1);
    return placa;
}

/** Pone `valor` en un disco de radio `r` (sin tocar el borde de la placa). */
export function estampar(placa: Float32Array, N: number, cx: number, cy: number, r: number, valor: number): void {
    for (let i = Math.max(1, Math.floor(cy - r)); i <= Math.min(N - 2, Math.ceil(cy + r)); i++) {
        for (let j = Math.max(1, Math.floor(cx - r)); j <= Math.min(N - 2, Math.ceil(cx + r)); j++) {
            if ((i - cy) ** 2 + (j - cx) ** 2 <= r * r) placa[i * N + j] = valor;
        }
    }
}

/** Posición de la fuente de calor móvil en el cuadro `t`: una órbita alrededor del centro. */
export const fuenteCalor = (N: number, t: number) => ({ cx: N / 2 + 0.3 * N * Math.cos(t / 60), cy: N / 2 + 0.3 * N * Math.sin(t / 45) });

async function simularCalor(dispositivo: Dispositivo, N: number): Promise<Simulacion> {
    const { device } = dispositivo;
    const { kernel, preparado } = await preparar(dispositivo, 'calor');
    const { alfa, pasosPorCuadro } = CALOR;
    const campo = placaInicial(N);
    const par: [string, string] = ['actual', 'siguiente'];
    const gpu = crearBuffers(device, kernel, { params: { N, alfa }, buffers: { actual: campo, siguiente: campo } });
    let cuadro = 0;
    return {
        inicio: { meta: { tipo: 'calor', N, pasosPorCuadro }, payload: new Uint8Array(0) },
        async siguiente() {
            // Una fuente caliente que se mueve mantiene viva la simulación; se escribe sobre la copia de la CPU y se sube.
            const { cx, cy } = fuenteCalor(N, cuadro++);
            estampar(campo, N, cx, cy, N * 0.03, 1);
            device.queue.writeBuffer(gpu.buffers.get('actual')!, 0, campo);
            const r = await lanzarKernel(device, preparado, gpu, { N, alfa }, { timestamps: false, lanzamientos: pasosPorCuadro, intercambio: par });
            const inicioLectura = performance.now();
            campo.set(new Float32Array(await leerBytes(device, gpu.buffers.get(bufferFinal(par, pasosPorCuadro))!)));
            const msLectura = performance.now() - inicioLectura;
            const payload = Uint8Array.from(campo, (v) => Math.min(255, Math.max(0, Math.round(v * 255))));
            return { meta: { tipo: 'cuadro', ms: r.msEspera, msLectura, trabajo: pasosPorCuadro * (N - 2) * (N - 2), unidad: 'celdas' }, payload };
        },
        destruir: () => gpu.destruir(),
    };
}

// ---------- multiplicación de matrices ----------

/** A: una imagen fija (anillos y rayas) con valores en [0, 1]. */
export function imagenA(N: number): Float32Array {
    return Float32Array.from({ length: N * N }, (_, k) => {
        const i = Math.floor(k / N) / N, j = (k % N) / N;
        const anillos = 0.5 + 0.5 * Math.sin(40 * Math.hypot(i - 0.5, j - 0.5));
        const rayas = 0.5 + 0.5 * Math.sin(18 * (i + j));
        return 0.6 * anillos + 0.4 * rayas;
    });
}

/**
 * B: matriz circulante que desplaza `desplazamiento` columnas y promedia `2·ancho + 1` vecinas.
 * Multiplicar A·B equivale a mover y desenfocar la imagen A hacia los lados.
 */
export function matrizB(N: number, desplazamiento: number, ancho: number, destino = new Float32Array(N * N)): Float32Array {
    destino.fill(0);
    const peso = 1 / (2 * ancho + 1);
    for (let k = 0; k < N; k++) {
        for (let d = -ancho; d <= ancho; d++) destino[k * N + ((((k + desplazamiento + d) % N) + N) % N)]! += peso;
    }
    return destino;
}

/** Parámetros de B en el cuadro `t`: avanza 3 columnas por cuadro y el desenfoque respira entre 0 y N/16. */
export const parametrosB = (N: number, t: number) => ({ desplazamiento: 3 * t, ancho: Math.round((N / 32) * (1 + Math.sin(t / 25))) });

const aU8 = (m: Float32Array) => Uint8Array.from(m, (v) => Math.min(255, Math.max(0, Math.round(v * 255))));

async function simularMatmul(dispositivo: Dispositivo, N: number): Promise<Simulacion> {
    const { device } = dispositivo;
    const { kernel, preparado } = await preparar(dispositivo, 'matmul');
    const A = imagenA(N);
    const B = new Float32Array(N * N);
    const gpu = crearBuffers(device, kernel, { params: { N }, buffers: { A, B: matrizB(N, 0, 0, B), C: N * N } });
    let cuadro = 0;
    return {
        inicio: { meta: { tipo: 'matmul', N }, payload: aU8(A) },
        async siguiente() {
            const { desplazamiento, ancho } = parametrosB(N, cuadro++);
            device.queue.writeBuffer(gpu.buffers.get('B')!, 0, matrizB(N, desplazamiento, ancho, B));
            const r = await lanzarKernel(device, preparado, gpu, { N }, { timestamps: false });
            const inicioLectura = performance.now();
            const C = new Float32Array(await leerBytes(device, gpu.buffers.get('C')!));
            const msLectura = performance.now() - inicioLectura;
            // el producto de A (en [0,1]) por una matriz de filas que suman 1 queda en [0,1]
            const payload = new Uint8Array(2 * N * N);
            payload.set(aU8(C), 0);
            payload.set(aU8(B), N * N);
            return { meta: { tipo: 'cuadro', ms: r.msEspera, msLectura, trabajo: 2 * N ** 3, unidad: 'operaciones', ancho }, payload };
        },
        destruir: () => gpu.destruir(),
    };
}

export async function crearSimulacion(dispositivo: Dispositivo, id: InfoSimulacion['id'], tam: number): Promise<Simulacion> {
    if (id === 'nbody') return simularNBody(dispositivo, tam);
    if (id === 'calor') return simularCalor(dispositivo, tam);
    return simularMatmul(dispositivo, tam);
}
