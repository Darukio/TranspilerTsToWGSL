import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TranspiladorTsaWGSL } from '../core/TranspiladorTSaWGSL.js';

/** Transpila un fragmento de TypeScript y devuelve lo que decidió el pipeline. */
function transpilarCodigo(cuerpo: string, firma = 'a: number[], b: number[], N: number, n: number') {
    const dir = mkdtempSync(join(tmpdir(), 'inyectividad-'));
    try {
        const archivo = join(dir, 'caso.ts');
        writeFileSync(archivo, `export function f(${firma}) {\n${cuerpo}\n}\n`);
        const ctx = new TranspiladorTsaWGSL().transpilar([archivo]);
        return { aceptado: ctx.kernels.length === 1, motivos: ctx.diagnosticos.filter((d) => d.fase === 'middleend').map((d) => d.mensaje).join(' | '), ctx };
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}

const bucle1 = (escritura: string) => `for (let i = 0; i < n; i++) { ${escritura} }`;
const bucle2 = (escritura: string, cotaJ = 'N') => `for (let i = 0; i < N; i++) { for (let j = 0; j < ${cotaJ}; j++) { ${escritura} } }`;

const ACEPTADOS: [string, string][] = [
    ['identidad a[i]', bucle1('a[i] = b[i]!;')],
    ['con paso 2 y desplazamiento a[2*i + 1]', bucle1('a[2 * i + 1] = b[i]!;')],
    ['fila mayor a[i*N + j]', bucle2('a[i * N + j] = b[i * N + j]!;')],
    ['columna mayor a[j*N + i]', bucle2('a[j * N + i] = b[i * N + j]!;')],
    ['interior de un stencil a[(i+1)*N + (j+1)] con i,j < N-2', `for (let i = 0; i < N - 2; i++) { for (let j = 0; j < N - 2; j++) { a[(i + 1) * N + (j + 1)] = b[i * N + j]!; } }`],
    ['tres dimensiones a[(i*N + j)*N + k]', `for (let i = 0; i < N; i++) { for (let j = 0; j < N; j++) { for (let k = 0; k < N; k++) { a[(i * N + j) * N + k] = b[k]!; } } }`],
    ['inverso a[n - 1 - i]', bucle1('a[n - 1 - i] = b[i]!;')],
    ['relleno literal a[i*3 + j] con j < 3', bucle2('a[i * 3 + j] = b[j]!;', '3')],
    ['bucle con <= : a[i] con i <= n', `for (let i = 0; i <= n; i++) { a[i] = b[i]!; }`],
    ['local constante en el índice: const c = (i+1)*N + (j+1); a[c] (interior de un stencil)', `for (let i = 0; i < N - 2; i++) { for (let j = 0; j < N - 2; j++) { const c = (i + 1) * N + (j + 1); a[c] = b[c]!; } }`],
    ['local constante numérica: let s = 3; a[i + s] (equivale a a[i + 3])', `for (let i = 0; i < n; i++) { let s = 3; a[i + s] = b[i]!; }`],
    ['local constante simple: const k = 2*i; a[k + 1]', `for (let i = 0; i < n; i++) { const k = 2 * i; a[k + 1] = b[i]!; }`],
    ['iterador interno que no es de grid: a[i*N + k]', `for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < N; k++) { a[i * N + k] = s; } }`],
];

const RECHAZADOS: [string, string, RegExp][] = [
    ['suma de iteradores a[i + j] (colisiona: (1,2) y (2,1))', bucle2('a[i + j] = b[j]!;'), /inyectiv/],
    ['división a[i / 2] (colisiona: 2 y 3)', bucle1('a[i / 2] = b[i]!;'), /inyectiv/],
    ['multiplicado por cero a[i * 0]', bucle1('a[i * 0] = b[i]!;'), /inyectiv/],
    ['paso desconocido a[i * N] (N podría ser 0)', bucle1('a[i * N] = b[i]!;'), /inyectiv/],
    ['pasos solapados a[i*2 + j] con j < 3 (colisiona: (0,2) y (1,0))', bucle2('a[i * 2 + j] = b[j]!;', '3'), /inyectiv/],
    ['paso que no cubre el rango a[i*N + j] con j < M (M no relacionado con N)', bucle2('a[i * N + j] = b[j]!;', 'n'), /inyectiv/],
    ['dos iteradores con el mismo paso a[i*N + j*N]', bucle2('a[i * N + j * N] = b[j]!;'), /inyectiv/],
    ['producto de iteradores a[i * j]', bucle2('a[i * j] = b[j]!;'), /inyectiv/],
    ['dispersión por datos a[b[i]]', bucle1('a[b[i]!] = 1;'), /inyectiv/],
    ['local reasignada en el índice: let c = i; c += 1; a[c] (no se sigue)', `for (let i = 0; i < n; i++) { let c = i; c += 1; a[c] = b[i]!; }`, /inyectiv/],
    ['local que viene de los datos: const k = b[i]; a[k]', `for (let i = 0; i < n; i++) { const k = b[i]!; a[k] = 1; }`, /inyectiv/],
    ['local constante que colisiona: const c = i + j; a[c]', bucle2('const c = i + j; a[c] = b[j]!;'), /inyectiv/],
    ['iterador interno que se solapa: a[i + k] con k < N', `for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < N; k++) { a[i + k] = s; } }`, /inyectiv/],
];

for (const [nombre, codigo] of ACEPTADOS) {
    test(`escritura inyectiva aceptada: ${nombre}`, () => {
        const r = transpilarCodigo(codigo);
        assert.ok(r.aceptado, `debió aceptarse; diagnósticos: ${r.motivos}`);
    });
}

for (const [nombre, codigo, motivo] of RECHAZADOS) {
    test(`escritura con posible colisión rechazada: ${nombre}`, () => {
        const r = transpilarCodigo(codigo);
        assert.equal(r.aceptado, false, 'una escritura que puede colisionar no debe volverse kernel');
        assert.match(r.motivos, motivo);
    });
}
