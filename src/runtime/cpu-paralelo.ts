import { Worker } from 'node:worker_threads';
import os from 'node:os';
import type { ConfigParalelo } from './cpu-worker.js';

/**
 * Línea base de CPU multihilo: el mismo código TypeScript (ya compilado) repartido entre hilos que comparten los
 * arrays por SharedArrayBuffer. Los hilos se crean una sola vez y se reutilizan en cada medición.
 */
export class PoolCPU {
    private vistas: Record<string, Float64Array> = {};

    private constructor(private readonly workers: Worker[]) { }

    static crear(config: ConfigParalelo, hilos: number = os.availableParallelism()): PoolCPU {
        const workers = Array.from({ length: hilos }, () => new Worker(new URL('./cpu-worker.js', import.meta.url), { workerData: config }));
        return new PoolCPU(workers);
    }

    get hilos(): number {
        return this.workers.length;
    }

    /** Envía un mensaje a cada hilo y espera a que todos respondan. */
    private tanda(mensaje: (k: number) => object): Promise<void> {
        return Promise.all(
            this.workers.map(
                (w, k) =>
                    new Promise<void>((ok, mal) => {
                        const alError = (e: Error) => { w.off('message', alMensaje); mal(e); };
                        const alMensaje = () => { w.off('error', alError); ok(); };
                        w.once('message', alMensaje);
                        w.once('error', alError);
                        w.postMessage(mensaje(k));
                    }),
            ),
        ).then(() => undefined);
    }

    /** Copia los arrays a memoria compartida (una vez por tamaño). Las salidas se pasan en cero. */
    async preparar(arrays: Record<string, Float64Array>, params: Record<string, number>): Promise<void> {
        const buffers: Record<string, SharedArrayBuffer> = {};
        this.vistas = {};
        for (const [nombre, datos] of Object.entries(arrays)) {
            const sab = new SharedArrayBuffer(datos.byteLength);
            const vista = new Float64Array(sab);
            vista.set(datos);
            buffers[nombre] = sab;
            this.vistas[nombre] = vista;
        }
        await this.tanda(() => ({ tipo: 'preparar', buffers, params }));
    }

    /** Reparte `unidades` (elementos o filas) en tramos parejos, uno por hilo, y espera a que terminen. */
    correr(unidades: number): Promise<void> {
        const w = this.workers.length;
        return this.tanda((k) => ({ tipo: 'correr', desde: Math.floor((unidades * k) / w), hasta: Math.floor((unidades * (k + 1)) / w) }));
    }

    leer(nombre: string): Float64Array {
        const vista = this.vistas[nombre];
        if (!vista) throw new Error(`No hay array "${nombre}" en el pool`);
        return vista;
    }

    async cerrar(): Promise<void> {
        await Promise.all(this.workers.map((w) => w.terminate()));
    }
}
