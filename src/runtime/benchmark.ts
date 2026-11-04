import { fileURLToPath } from 'node:url';
import { TranspiladorTsaWGSL } from '../core/TranspiladorTSaWGSL.js';
import { crearBuffers } from './buffers.js';
import { prepararKernel, lanzarKernel } from './dispatch.js';
import { leerSalidas } from './lectura.js';
import { PoolCPU } from './cpu-paralelo.js';
import type { ConfigParalelo } from './cpu-worker.js';
import type { Dispositivo } from './dispositivo.js';

export interface Estadisticas {
    mediana: number;
    minimo: number;
    maximo: number;
    media: number;
    muestras: number[];
}

export function estadisticas(muestras: number[]): Estadisticas {
    if (muestras.length === 0) throw new Error('No hay muestras');
    const orden = [...muestras].sort((a, b) => a - b);
    const mitad = Math.floor(orden.length / 2);
    const mediana = orden.length % 2 ? orden[mitad]! : (orden[mitad - 1]! + orden[mitad]!) / 2;
    return {
        mediana,
        minimo: orden[0]!,
        maximo: orden[orden.length - 1]!,
        media: muestras.reduce((a, b) => a + b, 0) / muestras.length,
        muestras,
    };
}

interface Datos {
    params: Record<string, number>;
    entradas: Record<string, Float64Array>;
    /** nombre -> largo en elementos */
    salidas: Record<string, number>;
}

export interface CasoBenchmark {
    id: string;
    titulo: string;
    /** Nombre del archivo en src/pruebas (sin extensión). */
    archivo: string;
    /** Función exportada que se ejecuta en la CPU como línea base. */
    funcion: string;
    /** Argumentos de la función, en orden: nombres de entradas, salidas o params. */
    orden: string[];
    /** Cómo se reparte entre hilos de CPU y qué parámetro da el tamaño (que es también el total de unidades a repartir). */
    paralelo: { modo: ConfigParalelo['modo']; parametro: string; /** Unidades a repartir entre hilos (por defecto, el valor del parámetro). */ unidades?: (params: Record<string, number>) => number };
    /** Tamaños del problema (el parámetro que escala). */
    tamanos: number[];
    /** Unidades de trabajo para el eje X del gráfico. */
    trabajo(tam: number): number;
    datos(tam: number): Datos;
}

const rellenar = (n: number, f: (i: number) => number) => Float64Array.from({ length: n }, (_, i) => f(i));

export const CASOS: CasoBenchmark[] = [
    {
        id: 'vectores', titulo: 'Vectores: salida[i] = A[i] * 3.14 - B[i]',
        archivo: 'vectores', funcion: 'procesarVectores', orden: ['vectorA', 'vectorB', 'salida', 'n'],
        paralelo: { modo: 'lineal', parametro: 'n' },
        tamanos: [1_000, 10_000, 100_000, 1_000_000, 4_000_000],
        trabajo: (n) => n,
        datos: (n) => ({ params: { n }, entradas: { vectorA: rellenar(n, (i) => (i % 1000) * 0.5), vectorB: rellenar(n, (i) => i % 7) }, salidas: { salida: n } }),
    },
    {
        id: 'matrices', titulo: 'Matrices N×N: C[i,j] = A[i,j] + B[i,j]',
        archivo: 'matrices', funcion: 'multiplicarMatrices', orden: ['A', 'B', 'C', 'N'],
        paralelo: { modo: 'filas', parametro: 'N' },
        tamanos: [32, 128, 256, 512, 1024],
        trabajo: (N) => N * N,
        datos: (N) => ({ params: { N }, entradas: { A: rellenar(N * N, (i) => i % 1000), B: rellenar(N * N, (i) => 1000 - (i % 1000)) }, salidas: { C: N * N } }),
    },
    {
        id: 'matmul', titulo: 'Multiplicación de matrices real N×N: C = A · B (2·N³ operaciones)',
        archivo: 'matmul', funcion: 'multiplicarMatricesReal', orden: ['A', 'B', 'C', 'N'],
        paralelo: { modo: 'filas', parametro: 'N' },
        tamanos: [64, 128, 256, 512, 1024, 2048],
        trabajo: (N) => N * N,
        datos: (N) => ({ params: { N }, entradas: { A: rellenar(N * N, (i) => ((i * 7) % 13) / 13), B: rellenar(N * N, (i) => ((i * 5) % 11) / 11) }, salidas: { C: N * N } }),
    },
    {
        id: 'nbody', titulo: 'N cuerpos (gravedad 2D): cada cuerpo suma la fuerza de todos los demás, O(n²)',
        archivo: 'nbody', funcion: 'calcularFuerzas', orden: ['x', 'y', 'masa', 'ax', 'ay', 'n', 'suavizado'],
        paralelo: { modo: 'filas', parametro: 'n' },
        // 16384² = 268 M interacciones por pasada: más allá, un solo envío se acercaría al límite de ~2 s de la GPU en Windows.
        tamanos: [256, 1_024, 4_096, 16_384],
        trabajo: (n) => n,
        datos: (n) => ({
            params: { n, suavizado: 0.01 },
            entradas: { x: rellenar(n, (i) => ((i * 7919) % 10007) / 100), y: rellenar(n, (i) => ((i * 104729) % 10009) / 100), masa: rellenar(n, (i) => 1 + (i % 5)) },
            salidas: { ax: n, ay: n },
        }),
    },
    {
        id: 'calor', titulo: 'Difusión de calor N×N: un paso de la ecuación del calor (stencil de 5 puntos, doble búfer)',
        archivo: 'calor', funcion: 'difundirCalor', orden: ['actual', 'siguiente', 'N', 'alfa'],
        paralelo: { modo: 'filas', parametro: 'N', unidades: (p) => p['N']! - 2 },
        tamanos: [128, 256, 512, 1_024, 2_048],
        trabajo: (N) => N * N,
        datos: (N) => ({ params: { N, alfa: 0.1 }, entradas: { actual: rellenar(N * N, (i) => ((i * 31) % 100) / 100) }, salidas: { siguiente: N * N } }),
    },
    {
        id: 'monte-carlo', titulo: 'Monte Carlo: sqrt(p * 2.5) + sin(p)',
        archivo: 'monte-carlo', funcion: 'simularMonteCarlo', orden: ['precios', 'resultados', 'n'],
        paralelo: { modo: 'lineal', parametro: 'n' },
        tamanos: [1_000, 10_000, 100_000, 1_000_000, 4_000_000],
        trabajo: (n) => n,
        datos: (n) => ({ params: { n }, entradas: { precios: rellenar(n, (i) => 1 + (i % 1000) * 0.01) }, salidas: { resultados: n } }),
    },
];

/** GPU con los datos residentes: subir una vez, lanzar `k` veces seguidas, leer una vez. */
export interface Amortizado {
    k: number;
    gpuTotal: Estadisticas;
    /** k pasadas de la mejor CPU (k × la mediana de una pasada). */
    cpu: number;
}

export interface Ajuste {
    /** Costo que no depende de k: subida + lectura + sincronización. */
    fijoMs: number;
    /** Costo de cada lanzamiento adicional. */
    porLanzamientoMs: number;
    /** Menor k a partir del cual la GPU gana el total; null si nunca (cada lanzamiento cuesta más que una pasada de CPU). */
    kEmpate: number | null;
}

/**
 * Ajusta total(k) ≈ fijo + k × porLanzamiento con el primer y el último punto medidos, y estima desde qué k
 * la GPU supera a la CPU. Es una estimación lineal: necesita al menos dos valores de k.
 */
export function ajusteAmortizado(puntos: { k: number; ms: number }[], cpuPorPasadaMs: number): Ajuste | undefined {
    if (puntos.length < 2) return undefined;
    const primero = puntos[0]!, ultimo = puntos[puntos.length - 1]!;
    const porLanzamientoMs = Math.max((ultimo.ms - primero.ms) / (ultimo.k - primero.k), 0);
    const fijoMs = Math.max(primero.ms - porLanzamientoMs * primero.k, 0);
    const kEmpate = cpuPorPasadaMs > porLanzamientoMs ? Math.max(1, Math.ceil(fijoMs / (cpuPorPasadaMs - porLanzamientoMs))) : null;
    return { fijoMs, porLanzamientoMs, kEmpate };
}

/** Por debajo de esto, la pendiente de pared es ruido y no se usa para decidir. */
const PARED_FIABLE_MS = 0.05;

/**
 * Compara el cronómetro de la GPU (timestamp-query) con el costo por lanzamiento medido con reloj de pared.
 * Cuando el de pared es fiable, se usa el MAYOR de los dos (el valor prudente) para no favorecer a la GPU; `divergencia`
 * es pared / cronómetro. Si el de pared no es fiable, queda el cronómetro.
 */
export function computePrudente(cronometroMs: number | undefined, paredMs: number | undefined): { ms: number; divergencia?: number } | undefined {
    const paredFiable = paredMs !== undefined && paredMs >= PARED_FIABLE_MS;
    if (cronometroMs === undefined) return paredFiable ? { ms: paredMs } : undefined;
    if (!paredFiable) return { ms: cronometroMs };
    return { ms: Math.max(cronometroMs, paredMs), divergencia: paredMs / cronometroMs };
}

/** ¿El cronómetro y el reloj de pared discrepan al menos 2 veces (en cualquier sentido)? */
export const hayDivergencia = (divergencia: number | undefined): boolean => divergencia !== undefined && (divergencia >= 2 || divergencia <= 0.5);

export interface ResultadoTamano {
    tam: number;
    trabajo: number;
    /** La función original, en un solo hilo. */
    cpu: Estadisticas;
    /** El mismo trabajo repartido entre todos los hilos de CPU. */
    cpuMulti: Estadisticas;
    /** Mediana de la CPU más rápida de las dos para este tamaño: contra ella compite la GPU. */
    cpuMejorMs: number;
    /** Subida + espera + lectura, de punta a punta (sin instrumentar con timestamps). */
    gpuTotal: Estadisticas;
    /** Solo el cómputo, medido por la GPU (timestamp-query). */
    gpuCompute?: Estadisticas;
    subida: Estadisticas;
    espera: Estadisticas;
    lectura: Estadisticas;
    /** Mayor error relativo entre la GPU (f32) y la CPU (f64). */
    errorMax: number;
    /** El mismo problema con los datos quedándose en la GPU durante k lanzamientos. */
    amortizado: Amortizado[];
    /** Ajuste del modo amortizado contra la mejor CPU. */
    ajuste?: Ajuste;
    /** Costo de cómputo por lanzamiento que se usa para decidir quién gana el cómputo: el mayor entre cronómetro y pared (si el de pared es fiable). */
    computePrudenteMs?: number;
    /** Tiempo de pared por lanzamiento: la pendiente medida (si hay ajuste) o, si no, espera − latencia fija. */
    paredPorLanzamientoMs?: number;
    /** Tiempo de pared por lanzamiento / cronómetro de la GPU. Lejos de 1 significa que el cronómetro no es confiable para este kernel. */
    divergenciaCronometro?: number;
}

export interface ResultadoCaso {
    caso: { id: string; titulo: string };
    msCompilacion: number;
    repeticiones: number;
    calentamiento: number;
    /** Hilos de la CPU multihilo. */
    hilosCpu: number;
    /** Optimizaciones que el emisor aplicó al kernel de la GPU (vacío = kernel ingenuo). */
    optimizaciones: string[];
    tamanos: ResultadoTamano[];
}

export interface Progreso {
    mensaje: string;
    /** Avance de este caso, de 0 a 1. */
    fraccion: number;
}

export interface OpcionesBenchmark {
    repeticiones: number;
    calentamiento: number;
    /** Descarta los tamaños mayores a este valor (útil para pruebas rápidas). */
    maxTam?: number;
    /** Sin k=1000 en el modo amortizado. */
    rapido?: boolean;
    /** Se revisa entre iteraciones: un envío que ya está en la GPU no se puede abortar. */
    signal?: AbortSignal;
    alProgreso?: (progreso: Progreso) => void;
}

// Valores de k a probar, y el tope de un envío a la GPU: Windows mata (TDR) los que pasan de ~2 s.
const KS = [1, 10, 100, 1000];
const KS_RAPIDO = [1, 10, 100];
const TOPE_MS_POR_ENVIO = 800;
// La CPU necesita al menos esto de calentamiento: el JIT de V8 tarda en optimizar y la dejaría en desventaja.
const CALENTAMIENTO_MIN_CPU = 5;
// Si la primera corrida de la CPU pasa de esto, esa serie usa 1 calentamiento y 3 repeticiones (si no, 20 corridas de
// varios segundos harían el benchmark interminable). El tooltip de la celda muestra cuántas muestras tuvo.
const CPU_LENTA_MS = 400;
const REPS_CPU_LENTA = 3;

const SRC_PRUEBAS = (archivo: string) => fileURLToPath(new URL(`../../src/pruebas/${archivo}.ts`, import.meta.url));

/** Etapas de cada tamaño y qué parte del avance ocupan (suman 1). */
const ETAPAS = {
    cpu: { desde: 0, peso: 0.08 },
    cpuMulti: { desde: 0.08, peso: 0.08 },
    gpuTotal: { desde: 0.16, peso: 0.24 },
    gpuCompute: { desde: 0.4, peso: 0.2 },
    amortizado: { desde: 0.6, peso: 0.4 },
} as const;

export async function ejecutarBenchmark(dispositivo: Dispositivo, caso: CasoBenchmark, opciones: OpcionesBenchmark): Promise<ResultadoCaso> {
    const { device } = dispositivo;
    const { repeticiones, calentamiento } = opciones;
    const calCPU = Math.max(calentamiento, CALENTAMIENTO_MIN_CPU);

    const ctx = new TranspiladorTsaWGSL().transpilar([SRC_PRUEBAS(caso.archivo)]);
    const kernel = ctx.kernels[0];
    const shader = ctx.shaders[0];
    if (!kernel || !shader) throw new Error(`No se pudo transpilar "${caso.archivo}": ${JSON.stringify(ctx.diagnosticos)}`);

    // La línea base es el mismo código TypeScript, ya compilado a JS por tsc.
    const modulo = (await import(new URL(`../pruebas/${caso.archivo}.js`, import.meta.url).href)) as Record<string, (...a: unknown[]) => void>;
    const cpuFn = modulo[caso.funcion];
    if (!cpuFn) throw new Error(`"${caso.funcion}" no está exportada por ${caso.archivo}`);

    const preparado = await prepararKernel(device, kernel, shader.wgsl);
    const pool = PoolCPU.crear({ modo: caso.paralelo.modo, archivo: caso.archivo, funcion: caso.funcion, orden: caso.orden, parametro: caso.paralelo.parametro });
    const resultados: ResultadoTamano[] = [];
    // La espera de un kernel casi vacío es, en la práctica, la latencia fija de sincronizar con la GPU.
    let latenciaFijaMs = Infinity;

    try {
        const tamanos = caso.tamanos.filter((t) => t <= (opciones.maxTam ?? Infinity));
        const ksPosibles = opciones.rapido ? KS_RAPIDO : KS;
        const revisar = () => opciones.signal?.throwIfAborted();
        const avisar = (indice: number, etapa: { desde: number; peso: number }, hecho: number, texto: string) =>
            opciones.alProgreso?.({
                mensaje: `${caso.id} · tamaño ${tamanos[indice]!.toLocaleString('es-AR')} · ${texto}`,
                fraccion: (indice + etapa.desde + etapa.peso * hecho) / tamanos.length,
            });
        revisar();
        opciones.alProgreso?.({ mensaje: `${caso.id} · shader compilado en ${preparado.msCompilacion.toFixed(0)} ms`, fraccion: 0 });

        for (const [indice, tam] of tamanos.entries()) {
            revisar();
            const datos = caso.datos(tam);
            const entradasF32 = Object.fromEntries(Object.entries(datos.entradas).map(([k, v]) => [k, Float32Array.from(v)]));
            const salidasCPU = Object.fromEntries(Object.entries(datos.salidas).map(([k, n]) => [k, new Float64Array(n)]));
            const argsCPU = caso.orden.map((nombre) => datos.entradas[nombre] ?? salidasCPU[nombre] ?? datos.params[nombre]);

            // Mide `correr` con el calentamiento de la CPU; si es muy lento, reduce las repeticiones (ver CPU_LENTA_MS).
            const medirCPU = async (etapa: { desde: number; peso: number }, etiqueta: string, correr: () => void | Promise<void>): Promise<number[]> => {
                let cal = calCPU, reps = repeticiones;
                const muestras: number[] = [];
                for (let i = 0; i < cal + reps; i++) {
                    revisar();
                    avisar(indice, etapa, i / (cal + reps), `${etiqueta}, corrida ${i + 1}/${cal + reps}`);
                    const t0 = performance.now();
                    await correr();
                    const dt = performance.now() - t0;
                    if (i === 0 && dt > CPU_LENTA_MS) { cal = 1; reps = Math.min(repeticiones, REPS_CPU_LENTA); }
                    if (i >= cal) muestras.push(dt);
                }
                return muestras;
            };

            // ---- Fase 1: la CPU sola (sin trabajo de GPU en paralelo que interfiera) ----
            const cpu = await medirCPU(ETAPAS.cpu, 'CPU 1 hilo', () => cpuFn(...argsCPU));

            // ---- Fase 2: la CPU con todos los hilos, y comprobación de que calcula lo mismo ----
            await pool.preparar({ ...datos.entradas, ...Object.fromEntries(Object.entries(datos.salidas).map(([k, n]) => [k, new Float64Array(n)])) }, datos.params);
            const cpuMulti = await medirCPU(ETAPAS.cpuMulti, `CPU ${pool.hilos} hilos`, () => pool.correr(caso.paralelo.unidades?.(datos.params) ?? datos.params[caso.paralelo.parametro]!));
            for (const [nombre, esperado] of Object.entries(salidasCPU)) {
                const obtenido = pool.leer(nombre);
                for (let k = 0; k < esperado.length; k++) {
                    if (obtenido[k] !== esperado[k]) {
                        throw new Error(`La CPU multihilo no coincide con la original en "${nombre}[${k}]": ${obtenido[k]} contra ${esperado[k]}. La línea base no es válida.`);
                    }
                }
            }

            // ---- Fase 3: la GPU de punta a punta (subida + cómputo + lectura), sin cronómetros internos ----
            const gpuTotal: number[] = [], subida: number[] = [], espera: number[] = [], lectura: number[] = [], compute: number[] = [];
            const total = calentamiento + repeticiones;
            let errorMax = 0;
            for (let i = 0; i < total; i++) {
                revisar();
                avisar(indice, ETAPAS.gpuTotal, i / total, `GPU de punta a punta, corrida ${i + 1}/${total}${i < calentamiento ? ' (calentamiento)' : ''}`);
                const t0 = performance.now();
                const gpu = crearBuffers(device, kernel, { params: datos.params, buffers: { ...entradasF32, ...datos.salidas } });
                await device.queue.onSubmittedWorkDone(); // que la subida termine de verdad antes de medirla
                const t1 = performance.now();
                try {
                    const r = await lanzarKernel(device, preparado, gpu, datos.params, { timestamps: false });
                    const t2 = performance.now();
                    const salidas = await leerSalidas(device, kernel, gpu);
                    const t3 = performance.now();

                    if (i >= calentamiento) {
                        subida.push(t1 - t0);
                        espera.push(r.msEspera);
                        lectura.push(t3 - t2);
                        gpuTotal.push(t3 - t0);
                    }
                    if (i === total - 1) {
                        for (const [nombre, esperado] of Object.entries(salidasCPU)) {
                            const real = salidas[nombre]!;
                            for (let k = 0; k < esperado.length; k++) {
                                errorMax = Math.max(errorMax, Math.abs(real[k]! - esperado[k]!) / Math.max(1, Math.abs(esperado[k]!)));
                            }
                        }
                    }
                } finally {
                    gpu.destruir();
                }
            }

            // ---- Fase 4: cómputo puro, en corridas aparte: leer el cronómetro de la GPU cuesta una lectura extra
            // y contaminaría el tiempo total ----
            if (dispositivo.timestampQuery) {
                const gpu = crearBuffers(device, kernel, { params: datos.params, buffers: { ...entradasF32, ...datos.salidas } });
                try {
                    for (let i = 0; i < total; i++) {
                        revisar();
                        avisar(indice, ETAPAS.gpuCompute, i / total, `cómputo de la GPU, corrida ${i + 1}/${total}`);
                        const r = await lanzarKernel(device, preparado, gpu, datos.params, { timestamps: true });
                        if (i >= calentamiento && r.msCompute !== undefined) compute.push(r.msCompute);
                    }
                } finally {
                    gpu.destruir();
                }
            }

            // ---- Fase 5: datos residentes: subir una vez, k lanzamientos en un solo envío, leer una vez ----
            const cpuMejorMs = Math.min(estadisticas(cpu).mediana, estadisticas(cpuMulti).mediana);
            const amortizado: Amortizado[] = [];
            // Tope de seguridad por envío (TDR de Windows). Se predice con el tiempo de PARED del k anterior, no con el
            // cronómetro de la GPU: en algunos kernels el cronómetro subestima varias veces el tiempo real.
            let previo: { k: number; ms: number } | undefined;
            for (const [posK, k] of ksPosibles.entries()) {
                if (previo && previo.ms * (k / previo.k) > TOPE_MS_POR_ENVIO) break;
                const muestras: number[] = [];
                const calA = Math.min(calentamiento, 2), repsA = Math.min(repeticiones, 7);
                for (let i = 0; i < calA + repsA; i++) {
                    revisar();
                    avisar(indice, ETAPAS.amortizado, (posK + i / (calA + repsA)) / ksPosibles.length, `datos en la GPU, k=${k} (${posK + 1}/${ksPosibles.length}), corrida ${i + 1}/${calA + repsA}`);
                    const t0 = performance.now();
                    const gpu = crearBuffers(device, kernel, { params: datos.params, buffers: { ...entradasF32, ...datos.salidas } });
                    try {
                        await device.queue.onSubmittedWorkDone();
                        await lanzarKernel(device, preparado, gpu, datos.params, { timestamps: false, lanzamientos: k });
                        await leerSalidas(device, kernel, gpu);
                    } finally {
                        gpu.destruir();
                    }
                    if (i >= calA) muestras.push(performance.now() - t0);
                }
                const estad = estadisticas(muestras);
                amortizado.push({ k, gpuTotal: estad, cpu: k * cpuMejorMs });
                previo = { k, ms: estad.mediana };
            }
            const ajuste = ajusteAmortizado(amortizado.map((a) => ({ k: a.k, ms: a.gpuTotal.mediana })), cpuMejorMs);

            // Tiempo de pared por lanzamiento: la pendiente medida si existe; si el tope de seguridad dejó solo k=1 (kernels
            // lentos), se estima como espera − latencia fija.
            latenciaFijaMs = Math.min(latenciaFijaMs, estadisticas(espera).mediana);
            const paredMs = ajuste?.porLanzamientoMs ?? Math.max(estadisticas(espera).mediana - latenciaFijaMs, 0);
            const prudente = computePrudente(compute.length > 0 ? estadisticas(compute).mediana : undefined, paredMs);
            resultados.push({
                tam, trabajo: caso.trabajo(tam), errorMax, cpuMejorMs, amortizado, ...(ajuste ? { ajuste } : {}),
                paredPorLanzamientoMs: paredMs,
                ...(prudente ? { computePrudenteMs: prudente.ms } : {}),
                ...(prudente?.divergencia !== undefined ? { divergenciaCronometro: prudente.divergencia } : {}),
                cpu: estadisticas(cpu), cpuMulti: estadisticas(cpuMulti), gpuTotal: estadisticas(gpuTotal),
                ...(compute.length > 0 ? { gpuCompute: estadisticas(compute) } : {}),
                subida: estadisticas(subida), espera: estadisticas(espera), lectura: estadisticas(lectura),
            });
        }
    } finally {
        await pool.cerrar();
    }

    return { caso: { id: caso.id, titulo: caso.titulo }, msCompilacion: preparado.msCompilacion, repeticiones, calentamiento, hilosCpu: pool.hilos, optimizaciones: shader.optimizaciones, tamanos: resultados };
}
