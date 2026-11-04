import type { KernelIR, Expr } from '../ir/ir-contract.js';
import { asignarEjes, calcularBindings, tamanoWorkgroup } from '../backend/Bindings.js';
import { USO, type BuffersGPU } from './buffers.js';
import { leerBytes } from './lectura.js';

const VISIBLE_EN_COMPUTE = 0x04; // GPUShaderStage.COMPUTE

export type Tres = [number, number, number];

export interface Lanzamiento {
    /** Workgroups lanzados por eje. */
    grupos: Tres;
    /** Hilos por workgroup (el mismo que escribió el emisor en el shader). */
    tamanoWorkgroup: Tres;
}

export interface ResultadoDispatch extends Lanzamiento {
    /** Tiempo medido por la propia GPU durante el pass de cómputo (timestamp-query). Solo si el dispositivo lo soporta. */
    msCompute?: number;
    /** Tiempo de pared desde `submit` hasta que la GPU avisa que terminó. */
    msEspera: number;
}

function evaluar(e: Expr, params: Record<string, number>): number {
    switch (e.op) {
        case 'lit':
            return e.value;
        case 'ref': {
            const valor = params[e.id];
            if (valor === undefined) throw new Error(`El tamaño del grid usa "${e.id}", que no tiene valor entre los parámetros`);
            return valor;
        }
        case 'cast':
            return evaluar(e.x, params);
        case 'bin': {
            const l = evaluar(e.l, params);
            const r = evaluar(e.r, params);
            if (e.operator === '+') return l + r;
            if (e.operator === '-') return l - r;
            if (e.operator === '*') return l * r;
            if (e.operator === '/') return Math.trunc(l / r);
            throw new Error(`Operador "${e.operator}" no soportado en el tamaño del grid`);
        }
        default:
            throw new Error(`Expresión "${e.op}" no soportada en el tamaño del grid`);
    }
}

/**
 * Hilos por workgroup que declara el propio shader. Es la fuente de verdad: así el lanzamiento no puede desincronizarse
 * de lo que el emisor escribió (por ejemplo, 16×16 con baldosas y 8×8 sin ellas).
 */
export function leerWorkgroup(wgsl: string): Tres | undefined {
    const m = /@workgroup_size\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/.exec(wgsl);
    return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : undefined;
}

/**
 * Cuántos workgroups hacen falta para cubrir el grid: ceil(tamaño / hilos por workgroup) en cada eje.
 * `hilosPorGrupo` es el tamaño que declara el shader; sin él se supone el tamaño estándar del emisor sin optimizaciones.
 */
export function calcularWorkgroups(kernel: KernelIR, params: Record<string, number> = {}, hilosPorGrupo?: Tres): Lanzamiento {
    const hilos = hilosPorGrupo ?? tamanoWorkgroup(kernel.grid.length);
    const grupos: Tres = [1, 1, 1];
    // El eje x puede no ser el primer iterador del grid: se respeta la asignación que usó el emisor.
    asignarEjes(kernel).forEach((indiceGrid, eje) => {
        const g = kernel.grid[indiceGrid]!;
        const tamano = evaluar(g.size, params);
        if (!Number.isInteger(tamano)) throw new Error(`El tamaño del grid en "${g.iter}" no es un entero (${tamano})`);
        grupos[eje] = Math.max(0, Math.ceil(tamano / hilos[eje]!));
    });
    return { grupos, tamanoWorkgroup: hilos };
}

export interface KernelPreparado {
    kernel: KernelIR;
    /** Hilos por workgroup según el shader. */
    workgroup: Tres | undefined;
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
    /** Lo que costó compilar el shader y crear el pipeline: se paga una vez, no por lanzamiento. */
    msCompilacion: number;
}

/** Corre `trabajo` dentro de un error scope de validación y convierte el error de WebGPU en una excepción. */
async function conValidacion<T>(device: GPUDevice, trabajo: () => Promise<T>): Promise<T> {
    device.pushErrorScope('validation');
    let resultado: T | undefined;
    let fallo: unknown;
    try {
        resultado = await trabajo();
    } catch (e) {
        fallo = e;
    }
    const error = await device.popErrorScope();
    if (fallo) throw fallo;
    if (error) throw new Error(`WebGPU: ${error.message}`);
    return resultado as T;
}

/** Compila el shader y crea el pipeline. Hacerlo una vez y lanzar muchas veces con `lanzarKernel`. */
export function prepararKernel(device: GPUDevice, kernel: KernelIR, wgsl: string): Promise<KernelPreparado> {
    return conValidacion(device, async () => {
        const inicio = performance.now();
        const modulo = device.createShaderModule({ code: wgsl, label: kernel.name });
        const compilacion = (await modulo.getCompilationInfo()).messages.filter((m) => m.type === 'error');
        if (compilacion.length > 0) {
            throw new Error(`El shader no compila:\n${compilacion.map((m) => `  ${m.lineNum}:${m.linePos} ${m.message}`).join('\n')}`);
        }

        // Layout explícito: con 'auto' WebGPU descarta los bindings que el shader no usa.
        const bindings = calcularBindings(kernel);
        const entradas: GPUBindGroupLayoutEntry[] = [];
        if (bindings.params) entradas.push({ binding: bindings.params.binding, visibility: VISIBLE_EN_COMPUTE, buffer: { type: 'uniform' } });
        for (const b of bindings.buffers) {
            entradas.push({ binding: b.binding, visibility: VISIBLE_EN_COMPUTE, buffer: { type: b.access === 'read' ? 'read-only-storage' : 'storage' } });
        }
        const bindGroupLayout = device.createBindGroupLayout({ entries: entradas });
        const pipeline = await device.createComputePipelineAsync({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: modulo, entryPoint: kernel.name },
        });
        return { kernel, workgroup: leerWorkgroup(wgsl), pipeline, bindGroupLayout, msCompilacion: performance.now() - inicio };
    });
}

/** Lanza un kernel ya preparado sobre los buffers `gpu`. Resuelve cuando la GPU terminó. */
export async function lanzarKernel(
    device: GPUDevice,
    preparado: KernelPreparado,
    gpu: BuffersGPU,
    params: Record<string, number> = {},
    opciones: { timestamps?: boolean; lanzamientos?: number; intercambio?: [string, string] } = {},
): Promise<ResultadoDispatch> {
    const { kernel } = preparado;
    // Varios lanzamientos seguidos sobre los mismos buffers, en un solo envío: modela datos que se quedan en la GPU.
    const veces = opciones.lanzamientos ?? 1;
    if (!Number.isInteger(veces) || veces < 1) throw new Error(`"lanzamientos" debe ser un entero positivo (recibido ${veces})`);
    const lanzamiento = calcularWorkgroups(kernel, params, preparado.workgroup);
    const maximo = device.limits.maxComputeWorkgroupsPerDimension;
    if (lanzamiento.grupos.some((g) => g > maximo)) {
        throw new Error(`El grid necesita ${lanzamiento.grupos.join('x')} workgroups y la GPU permite ${maximo} por eje`);
    }

    const bindings = calcularBindings(kernel);
    if (bindings.params && !gpu.params) throw new Error('El kernel declara parámetros pero no hay buffer de params');

    // Los timestamps son opcionales: solo si el dispositivo se abrió con `timestamp-query`. Leer el
    // cronómetro cuesta una lectura extra, así que se puede apagar cuando se mide el tiempo total.
    const medir = opciones.timestamps !== false && device.features.has('timestamp-query');
    const marcas = medir ? device.createQuerySet({ type: 'timestamp', count: 2 }) : undefined;
    const resuelto = medir ? device.createBuffer({ size: 16, usage: USO.QUERY_RESOLVE | USO.COPY_SRC, label: 'timestamps' }) : undefined;

    try {
        const msEspera = await conValidacion(device, async () => {
            // Con `intercambio: [a, b]` los dos buffers se turnan entre lanzamientos: en los pares el kernel lee `a` y escribe `b`,
            // en los impares al revés (doble búfer de una simulación). Los dos deben ser del mismo tamaño.
            const par = opciones.intercambio;
            if (par) {
                const [a, b] = par.map((n) => gpu.buffers.get(n));
                if (!a || !b) throw new Error(`"intercambio" nombra un buffer que el kernel no tiene: ${par.join(', ')}`);
                if (a.size !== b.size) throw new Error(`Los buffers "${par[0]}" y "${par[1]}" deben tener el mismo tamaño para intercambiarse (${a.size} y ${b.size} bytes)`);
            }
            const enlazar = (intercambiar: boolean): GPUBindGroup => {
                const recursos: GPUBindGroupEntry[] = [];
                if (bindings.params) recursos.push({ binding: bindings.params.binding, resource: { buffer: gpu.params! } });
                for (const b of bindings.buffers) {
                    const nombre = intercambiar && par ? (b.name === par[0] ? par[1] : b.name === par[1] ? par[0] : b.name) : b.name;
                    const buffer = gpu.buffers.get(nombre);
                    if (!buffer) throw new Error(`No hay buffer en la GPU para "${nombre}"`);
                    recursos.push({ binding: b.binding, resource: { buffer } });
                }
                return device.createBindGroup({ layout: preparado.bindGroupLayout, entries: recursos });
            };
            const bindGroup = enlazar(false);
            const bindGroupIntercambiado = par ? enlazar(true) : bindGroup;

            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass(
                marcas ? { timestampWrites: { querySet: marcas, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined,
            );
            pass.setPipeline(preparado.pipeline);
            for (let i = 0; i < veces; i++) {
                pass.setBindGroup(0, i % 2 === 0 ? bindGroup : bindGroupIntercambiado);
                pass.dispatchWorkgroups(...lanzamiento.grupos);
            }
            pass.end();
            if (marcas && resuelto) encoder.resolveQuerySet(marcas, 0, 2, resuelto, 0);
            const comandos = encoder.finish();

            const inicio = performance.now();
            device.queue.submit([comandos]);
            await device.queue.onSubmittedWorkDone();
            return performance.now() - inicio;
        });

        if (!resuelto) return { ...lanzamiento, msEspera };
        const [inicio, fin] = new BigUint64Array(await leerBytes(device, resuelto));
        return { ...lanzamiento, msEspera, msCompute: Number(fin! - inicio!) / 1e6 }; // los timestamps vienen en nanosegundos
    } finally {
        marcas?.destroy();
        resuelto?.destroy();
    }
}

/** Compila y lanza una vez. Para repetir lanzamientos usar `prepararKernel` + `lanzarKernel`. */
export async function ejecutarKernel(
    device: GPUDevice,
    kernel: KernelIR,
    wgsl: string,
    gpu: BuffersGPU,
    params: Record<string, number> = {},
): Promise<ResultadoDispatch> {
    return lanzarKernel(device, await prepararKernel(device, kernel, wgsl), gpu, params);
}

/** Tras `pasos` lanzamientos con `intercambio: [a, b]`, el estado más reciente está en `b` si los pasos son impares y en `a` si son pares. */
export function bufferFinal(intercambio: [string, string], pasos: number): string {
    return pasos % 2 === 1 ? intercambio[1] : intercambio[0];
}
