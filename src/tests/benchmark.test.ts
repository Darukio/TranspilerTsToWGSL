import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AddressInfo } from 'node:net';
import { request } from 'node:http';
import { CASOS, ajusteAmortizado, computePrudente, ejecutarBenchmark, estadisticas, hayDivergencia } from '../runtime/benchmark.js';
import { crearServidor } from '../server/servidor.js';
import { conGPU } from './ayudas-gpu.js';

test('estadisticas: mediana, mínimo, máximo y media', () => {
    const impar = estadisticas([5, 1, 3]);
    assert.deepEqual([impar.mediana, impar.minimo, impar.maximo, impar.media], [3, 1, 5, 3]);
    assert.equal(estadisticas([1, 2, 3, 4]).mediana, 2.5);
    assert.throws(() => estadisticas([]), /No hay muestras/);
});

// Tamaño máximo chico para que el test sea rápido, y la tolerancia que corresponde a cada caso.
const PRUEBAS = [
    { id: 'vectores', max: 10_000, tolerancia: 1e-5 },
    { id: 'matrices', max: 128, tolerancia: 1e-6 },
    { id: 'matmul', max: 128, tolerancia: 1e-5 },
    { id: 'nbody', max: 1_024, tolerancia: 1e-3 },
    { id: 'calor', max: 256, tolerancia: 1e-5 },
    { id: 'monte-carlo', max: 10_000, tolerancia: 1e-3 },
];

for (const { id, max, tolerancia } of PRUEBAS) {
    test(`benchmark ${id}: mide CPU y GPU y el resultado coincide`, async (t) => {
        const caso = CASOS.find((c) => c.id === id)!;
        await conGPU(t, async (dispositivo) => {
            const r = await ejecutarBenchmark(dispositivo, caso, { repeticiones: 2, calentamiento: 1, maxTam: max });
            assert.ok(r.tamanos.length >= 2);
            assert.ok(r.msCompilacion > 0);
            for (const x of r.tamanos) {
                const donde = `${id} n=${x.tam}`;
                assert.equal(x.cpu.muestras.length, 2, donde);
                assert.equal(x.cpuMulti.muestras.length, 2, donde);
                assert.equal(x.cpuMejorMs, Math.min(x.cpu.mediana, x.cpuMulti.mediana), donde);
                assert.ok(r.hilosCpu >= 1, donde);
                assert.equal(x.gpuTotal.muestras.length, 2, donde);
                assert.ok(x.cpu.mediana > 0 && x.gpuTotal.mediana > 0, `${donde}: cpu=${x.cpu.mediana} gpuTotal=${x.gpuTotal.mediana}`);
                assert.ok(x.errorMax < tolerancia, `${donde}: errorMax=${x.errorMax}`);
                assert.equal(x.gpuCompute !== undefined, dispositivo.timestampQuery, donde);
                // El tope de seguridad (envíos de más de ~0,8 s) puede cortar los k grandes si la máquina está cargada:
                // k=1 siempre se mide, y el ajuste solo existe con dos o más valores de k.
                assert.ok(x.paredPorLanzamientoMs !== undefined && x.paredPorLanzamientoMs >= 0, donde);
                assert.equal(x.amortizado[0]!.k, 1, donde);
                assert.equal(x.ajuste !== undefined, x.amortizado.length >= 2, `${donde}: k=${x.amortizado.map((a) => a.k)}`);
                if (x.ajuste) assert.ok(x.ajuste.kEmpate === null || x.ajuste.kEmpate >= 1, `${donde}: kEmpate=${x.ajuste.kEmpate}`);
                if (dispositivo.timestampQuery && x.ajuste && x.ajuste.porLanzamientoMs >= 0.05) assert.ok(x.computePrudenteMs! >= x.gpuCompute!.mediana, donde);
            }
        });
    });
}

const H = { 'X-Benchmark': '1' };

test('servidor: página, API, validación y un solo benchmark a la vez', async (t) => {
    await conGPU(t, async (dispositivo) => {
        const servidor = crearServidor(dispositivo);
        await new Promise<void>((ok) => servidor.listen(0, '127.0.0.1', ok));
        const base = `http://127.0.0.1:${(servidor.address() as AddressInfo).port}`;
        try {
            const pagina = await fetch(`${base}/`);
            assert.equal(pagina.status, 200);
            assert.match(pagina.headers.get('content-type') ?? '', /text\/html/);
            assert.match(await pagina.text(), /Benchmark CPU vs GPU/);

            const info = (await (await fetch(`${base}/api/dispositivo`)).json()) as { software: boolean; timestampQuery: boolean };
            assert.equal(info.software, dispositivo.software);
            assert.equal(((await (await fetch(`${base}/api/casos`)).json()) as unknown[]).length, CASOS.length);

            assert.equal((await fetch(`${base}/api/benchmark?caso=vectores`)).status, 403); // sin cabecera
            assert.equal((await fetch(`${base}/api/benchmark?caso=inexistente`, { headers: H })).status, 400);
            assert.equal((await fetch(`${base}/nada`)).status, 404);
            assert.equal((await fetch(`${base}/`, { method: 'POST' })).status, 405);

            const url = `${base}/api/benchmark?caso=vectores&reps=2&calentamiento=1&max=10000`;
            const [a, b] = await Promise.all([fetch(url, { headers: H }), fetch(url, { headers: H })]);
            assert.deepEqual([a.status, b.status].sort(), [200, 409]);
            const ok = (a.status === 200 ? a : b);
            const cuerpo = (await ok.json()) as { tamanos: { tam: number }[] };
            assert.deepEqual(cuerpo.tamanos.map((x) => x.tam), [1_000, 10_000]);
        } finally {
            await new Promise<void>((ok) => servidor.close(() => ok()));
        }
    });
});

test('ajusteAmortizado: costo fijo, costo por lanzamiento y k de empate', () => {
    // total(k) = 9 + 1·k  → con una pasada de CPU de 2 ms, la GPU gana desde k = 9
    const a = ajusteAmortizado([{ k: 1, ms: 10 }, { k: 101, ms: 110 }], 2)!;
    assert.deepEqual([a.fijoMs, a.porLanzamientoMs, a.kEmpate], [9, 1, 9]);
    // si cada lanzamiento cuesta más que la CPU, nunca empata
    assert.equal(ajusteAmortizado([{ k: 1, ms: 10 }, { k: 101, ms: 110 }], 1)!.kEmpate, null);
    assert.equal(ajusteAmortizado([{ k: 1, ms: 10 }], 2), undefined);
});

/** Lee un flujo NDJSON del servidor y llama a `alEvento` por cada línea. */
async function leerFlujo(respuesta: Response, alEvento: (e: { tipo: string; [k: string]: unknown }) => void | Promise<void>) {
    const lector = respuesta.body!.pipeThrough(new TextDecoderStream()).getReader();
    let resto = '';
    for (;;) {
        const { value, done } = await lector.read();
        if (done) return;
        const lineas = (resto + value).split('\n');
        resto = lineas.pop()!;
        for (const l of lineas.filter(Boolean)) await alEvento(JSON.parse(l));
    }
}

async function conServidor(t: { skip(m?: string): void }, prueba: (base: string) => Promise<void>) {
    await conGPU(t, async (dispositivo) => {
        const servidor = crearServidor(dispositivo);
        await new Promise<void>((ok) => servidor.listen(0, '127.0.0.1', ok));
        try { await prueba(`http://127.0.0.1:${(servidor.address() as AddressInfo).port}`); }
        finally { await new Promise<void>((ok) => servidor.close(() => ok())); }
    });
}

test('servidor: el flujo emite progreso creciente y termina con el resultado; modo rápido sin k=1000', async (t) => {
    await conServidor(t, async (base) => {
        const eventos: { tipo: string; fraccion?: number; mensaje?: string; resultado?: { tamanos: { amortizado: { k: number }[] }[] } }[] = [];
        const r = await fetch(`${base}/api/benchmark/stream?caso=vectores&max=100000&rapido=1`, { headers: H });
        assert.match(r.headers.get('content-type') ?? '', /ndjson/);
        await leerFlujo(r, (e) => void eventos.push(e));

        const progreso = eventos.filter((e) => e.tipo === 'progreso');
        assert.ok(progreso.length >= 5);
        assert.ok(progreso.every((p, i) => i === 0 || p.fraccion! >= progreso[i - 1]!.fraccion! - 1e-9), 'el avance no retrocede');
        assert.ok(progreso.every((p) => p.fraccion! >= 0 && p.fraccion! <= 1));
        assert.ok(progreso.some((p) => /datos en la GPU, k=10/.test(p.mensaje!)));
        const ultimo = eventos[eventos.length - 1]!;
        assert.equal(ultimo.tipo, 'resultado');
        const ks = ultimo.resultado!.tamanos.flatMap((x) => x.amortizado.map((a) => a.k));
        assert.ok(ks.length > 0 && Math.max(...ks) <= 100, `k máximo ${Math.max(...ks)}`);
    });
});

test('servidor: cancelar detiene la corrida, avisa y deja la GPU libre', async (t) => {
    await conServidor(t, async (base) => {
        const eventos: string[] = [];
        const r = await fetch(`${base}/api/benchmark/stream?caso=monte-carlo&reps=100&calentamiento=0`, { headers: H });
        let cancelando: Promise<Response> | undefined;
        await leerFlujo(r, (e) => {
            eventos.push(e.tipo);
            if (e.tipo === 'progreso' && !cancelando) cancelando = fetch(`${base}/api/cancelar`, { method: 'POST', headers: H });
        });
        assert.equal(eventos[eventos.length - 1], 'cancelado');
        assert.ok(!eventos.includes('resultado'));
        assert.deepEqual(await (await cancelando!).json(), { cancelado: true });

        // la GPU quedó libre: una corrida nueva no recibe 409
        assert.equal((await fetch(`${base}/api/benchmark?caso=vectores&reps=1&calentamiento=0&max=1000`, { headers: H })).status, 200);
        // sin nada en curso, cancelar es inofensivo
        assert.deepEqual(await (await fetch(`${base}/api/cancelar`, { method: 'POST', headers: H })).json(), { cancelado: false });
    });
});

test('servidor: rechaza hosts ajenos y cancelar sin cabecera ni POST', async (t) => {
    await conServidor(t, async (base) => {
        const { port } = new URL(base);
        const estado = await new Promise<number>((ok, mal) => {
            const req = request({ host: '127.0.0.1', port, path: '/api/dispositivo', headers: { Host: 'sitio-ajeno.example' } }, (res) => { res.resume(); ok(res.statusCode!); });
            req.on('error', mal);
            req.end();
        });
        assert.equal(estado, 403);
        assert.equal((await fetch(`${base}/api/cancelar`, { method: 'POST' })).status, 403);
        assert.equal((await fetch(`${base}/api/cancelar`, { headers: H })).status, 405);
    });
});

test('computePrudente: usa el mayor entre cronómetro y pared solo si la pared es fiable', () => {
    // el cronómetro dice 0.17 ms y la pared 0.70 ms: gana el valor prudente (mayor) y se informa la divergencia
    assert.deepEqual(computePrudente(0.17, 0.7), { ms: 0.7, divergencia: 0.7 / 0.17 });
    // la pared es menor: se conserva el cronómetro, que es el mayor
    assert.equal(computePrudente(0.5, 0.4)!.ms, 0.5);
    // pared por debajo del umbral de ruido: no se usa
    assert.deepEqual(computePrudente(0.007, 0.002), { ms: 0.007 });
    assert.deepEqual(computePrudente(undefined, 0.2), { ms: 0.2 });
    assert.equal(computePrudente(undefined, 0.01), undefined);
});

test('hayDivergencia: avisa desde 2 veces en cualquier sentido', () => {
    assert.deepEqual([undefined, 1, 1.9, 2, 4.1, 0.5, 0.6].map(hayDivergencia), [false, false, false, true, true, true, false]);
});
