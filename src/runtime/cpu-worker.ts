import { parentPort, workerData } from 'node:worker_threads';

export interface ConfigParalelo {
    /** 'lineal': la función original se aplica a un tramo de los arrays. 'filas': versión por filas escrita a mano. */
    modo: 'lineal' | 'filas';
    archivo: string;
    funcion: string;
    /** Argumentos de la función original, en orden. */
    orden: string[];
    /** Nombre del parámetro que da el tamaño (n o N). */
    parametro: string;
}

type Vistas = Record<string, Float64Array>;

// La función original de matrices recorre N × N y no se puede cortar por filas: esta versión hace lo mismo
// por un rango de filas. El benchmark comprueba que su resultado coincide con el de la original.
const POR_FILAS: Record<string, (v: Vistas, p: Record<string, number>, desde: number, hasta: number) => void> = {
    matrices: (v, p, desde, hasta) => {
        const N = p['N']!;
        for (let i = desde; i < hasta; i++) for (let j = 0; j < N; j++) v['C']![i * N + j] = v['A']![i * N + j]! + v['B']![i * N + j]!;
    },
};

// Multiplicación de matrices por filas, con bloqueo de caché: para cada tramo de k y de j se reutilizan las filas de B que
// caben en caché en lugar de recorrerlas enteras. Cada C[i][j] suma sus productos en el mismo orden de k que la función
// original (0, 1, 2, …), así que el resultado es idéntico bit a bit; el benchmark lo comprueba.
const BLOQUE_K = 64;
const BLOQUE_J = 256;
POR_FILAS['matmul'] = (v, p, desde, hasta) => {
    const N = p['N']!;
    const A = v['A']!, B = v['B']!, C = v['C']!;
    C.fill(0, desde * N, hasta * N);
    for (let kk = 0; kk < N; kk += BLOQUE_K) {
        const kMax = Math.min(kk + BLOQUE_K, N);
        for (let jj = 0; jj < N; jj += BLOQUE_J) {
            const jMax = Math.min(jj + BLOQUE_J, N);
            for (let i = desde; i < hasta; i++) {
                const fila = i * N;
                for (let k = kk; k < kMax; k++) {
                    const a = A[fila + k]!;
                    const filaB = k * N;
                    for (let j = jj; j < jMax; j++) C[fila + j] = C[fila + j]! + a * B[filaB + j]!;
                }
            }
        }
    }
};

// n-cuerpos: cada hilo calcula la aceleración de un tramo de cuerpos recorriendo todos los demás; mismas operaciones y mismo
// orden que la función original, así que el resultado es idéntico.
POR_FILAS['nbody'] = (v, p, desde, hasta) => {
    const x = v['x']!, y = v['y']!, masa = v['masa']!, ax = v['ax']!, ay = v['ay']!;
    const n = p['n']!, suavizado = p['suavizado']!;
    for (let i = desde; i < hasta; i++) {
        let fx = 0;
        let fy = 0;
        for (let j = 0; j < n; j++) {
            const dx = x[j]! - x[i]!;
            const dy = y[j]! - y[i]!;
            const d2 = dx * dx + dy * dy + suavizado;
            const inv = 1 / (d2 * Math.sqrt(d2));
            fx += masa[j]! * dx * inv;
            fy += masa[j]! * dy * inv;
        }
        ax[i] = fx;
        ay[i] = fy;
    }
};

// Difusión de calor: un tramo de filas del interior (la fila i del tramo es la fila i+1 de la placa).
POR_FILAS['calor'] = (v, p, desde, hasta) => {
    const actual = v['actual']!, siguiente = v['siguiente']!;
    const N = p['N']!, alfa = p['alfa']!;
    for (let i = desde; i < hasta; i++) {
        for (let j = 0; j < N - 2; j++) {
            const c = (i + 1) * N + (j + 1);
            siguiente[c] = actual[c]! + alfa * (actual[c - N]! + actual[c + N]! + actual[c - 1]! + actual[c + 1]! - 4 * actual[c]!);
        }
    }
};

const config = workerData as ConfigParalelo;
const modulo = (await import(new URL(`../pruebas/${config.archivo}.js`, import.meta.url).href)) as Record<string, (...args: unknown[]) => void>;
const original = modulo[config.funcion];
if (!original) throw new Error(`"${config.funcion}" no está exportada por ${config.archivo}`);

let vistas: Vistas = {};
let params: Record<string, number> = {};

parentPort!.on('message', (m: { tipo: string; buffers?: Record<string, SharedArrayBuffer>; params?: Record<string, number>; desde?: number; hasta?: number }) => {
    if (m.tipo === 'preparar') {
        vistas = Object.fromEntries(Object.entries(m.buffers!).map(([k, sab]) => [k, new Float64Array(sab)]));
        params = m.params!;
    } else if (config.modo === 'filas') {
        const porFilas = POR_FILAS[config.archivo];
        if (!porFilas) throw new Error(`No hay versión por filas de "${config.archivo}"`);
        porFilas(vistas, params, m.desde!, m.hasta!);
    } else {
        const largo = m.hasta! - m.desde!;
        original(...config.orden.map((nombre) => (nombre in vistas ? vistas[nombre]!.subarray(m.desde, m.hasta) : nombre === config.parametro ? largo : params[nombre])));
    }
    parentPort!.postMessage('listo');
});
