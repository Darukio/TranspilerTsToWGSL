import { createServer, type Server, type ServerResponse } from 'node:http';
import { readFile } from 'node:fs/promises';
import { CASOS, ejecutarBenchmark, type OpcionesBenchmark } from '../runtime/benchmark.js';
import { describirDispositivo, type Dispositivo } from '../runtime/dispositivo.js';

const PAGINA = new URL('../../web/index.html', import.meta.url);

// El servidor puede lanzar trabajo en la GPU: solo atiende a `localhost` (evita DNS rebinding) y los
// endpoints que lanzan o cancelan trabajo exigen esta cabecera, que una página ajena no puede enviar
// sin un permiso CORS que este servidor nunca concede.
const CABECERA = 'x-benchmark';
const HOST_PERMITIDO = /^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/;

function json(res: ServerResponse, estado: number, cuerpo: unknown): void {
    res.writeHead(estado, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(cuerpo));
}

/** Convierte un parámetro de query en entero dentro de [min, max]; valor por defecto si falta o no es válido. */
function entero(valor: string | null, defecto: number, min: number, max: number): number {
    const n = Number(valor);
    return valor !== null && Number.isInteger(n) ? Math.min(max, Math.max(min, n)) : defecto;
}

function opcionesDe(url: URL): Omit<OpcionesBenchmark, 'signal' | 'alProgreso'> {
    const rapido = url.searchParams.get('rapido') === '1';
    const maxTam = url.searchParams.get('max');
    return {
        repeticiones: entero(url.searchParams.get('reps'), rapido ? 5 : 15, 1, 100),
        calentamiento: entero(url.searchParams.get('calentamiento'), rapido ? 1 : 3, 0, 20),
        rapido,
        ...(maxTam !== null ? { maxTam: entero(maxTam, Infinity, 1, Number.MAX_SAFE_INTEGER) } : {}),
    };
}

interface Corrida {
    controlador: AbortController;
    terminada: Promise<void>;
}

/**
 * Sirve la página y una API mínima. Un solo benchmark a la vez: comparten la GPU y se falsearían los tiempos.
 * `/api/benchmark/stream` emite una línea JSON por evento (progreso, resultado, cancelado o error).
 */
export function crearServidor(dispositivo: Dispositivo): Server {
    let corrida: Corrida | undefined;

    /** Reserva la GPU para una corrida; devuelve undefined si ya hay otra en curso. */
    function iniciar(): { corrida: Corrida; liberar: () => void } | undefined {
        if (corrida) return undefined;
        let liberar = () => {};
        const terminada = new Promise<void>((ok) => (liberar = ok));
        const nueva: Corrida = { controlador: new AbortController(), terminada };
        corrida = nueva;
        return { corrida: nueva, liberar: () => { corrida = undefined; liberar(); } };
    }

    async function transmitir(res: ServerResponse, url: URL): Promise<void> {
        const caso = CASOS.find((c) => c.id === url.searchParams.get('caso'));
        if (!caso) return json(res, 400, { error: `Caso desconocido. Disponibles: ${CASOS.map((c) => c.id).join(', ')}` });
        const reserva = iniciar();
        if (!reserva) return json(res, 409, { error: 'Ya hay un benchmark en curso' });

        const { controlador } = reserva.corrida;
        res.writeHead(200, { 'Content-Type': 'application/x-ndjson; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
        const emitir = (evento: unknown) => { if (!res.writableEnded) res.write(`${JSON.stringify(evento)}\n`); };
        // Si el navegador cierra la conexión (pestaña cerrada o fetch abortado), se cancela la corrida.
        res.on('close', () => { if (!res.writableFinished) controlador.abort(); });

        try {
            const resultado = await ejecutarBenchmark(dispositivo, caso, {
                ...opcionesDe(url),
                signal: controlador.signal,
                alProgreso: (p) => emitir({ tipo: 'progreso', ...p }),
            });
            emitir({ tipo: 'resultado', resultado });
        } catch (error) {
            emitir(controlador.signal.aborted ? { tipo: 'cancelado' } : { tipo: 'error', mensaje: error instanceof Error ? error.message : String(error) });
        } finally {
            reserva.liberar();
            res.end();
        }
    }

    return createServer(async (req, res) => {
        try {
            if (!HOST_PERMITIDO.test(req.headers.host ?? '')) return json(res, 403, { error: 'Host no permitido' });
            const url = new URL(req.url ?? '/', 'http://localhost');
            const lanzaTrabajo = url.pathname.startsWith('/api/benchmark') || url.pathname === '/api/cancelar';
            if (lanzaTrabajo && !req.headers[CABECERA]) return json(res, 403, { error: `Falta la cabecera ${CABECERA}` });

            if (url.pathname === '/api/cancelar') {
                if (req.method !== 'POST') return json(res, 405, { error: 'Solo POST' });
                const actual = corrida;
                actual?.controlador.abort();
                await actual?.terminada; // responde cuando la GPU ya quedó libre
                return json(res, 200, { cancelado: actual !== undefined });
            }
            if (req.method !== 'GET') return json(res, 405, { error: 'Solo GET' });

            if (url.pathname === '/') {
                res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
                return void res.end(await readFile(PAGINA));
            }
            if (url.pathname === '/api/dispositivo') {
                return json(res, 200, {
                    descripcion: describirDispositivo(dispositivo),
                    info: dispositivo.info,
                    software: dispositivo.software,
                    timestampQuery: dispositivo.timestampQuery,
                });
            }
            if (url.pathname === '/api/casos') {
                return json(res, 200, CASOS.map(({ id, titulo, tamanos }) => ({ id, titulo, tamanos })));
            }
            if (url.pathname === '/api/benchmark/stream') return await transmitir(res, url);
            if (url.pathname === '/api/benchmark') {
                const caso = CASOS.find((c) => c.id === url.searchParams.get('caso'));
                if (!caso) return json(res, 400, { error: `Caso desconocido. Disponibles: ${CASOS.map((c) => c.id).join(', ')}` });
                const reserva = iniciar();
                if (!reserva) return json(res, 409, { error: 'Ya hay un benchmark en curso' });
                try {
                    return json(res, 200, await ejecutarBenchmark(dispositivo, caso, { ...opcionesDe(url), signal: reserva.corrida.controlador.signal }));
                } finally {
                    reserva.liberar();
                }
            }
            return json(res, 404, { error: 'No existe' });
        } catch (error) {
            if (!res.headersSent) return json(res, 500, { error: error instanceof Error ? error.message : String(error) });
            res.end();
        }
    });
}
