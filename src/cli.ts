#!/usr/bin/env node
// src/cli.ts
import { resolve, join } from 'node:path';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { TranspiladorTsaWGSL } from './core/TranspiladorTSaWGSL.js';

const AYUDA = `
Uso:
  ts-to-wgsl [-o <dir>] <archivo1.ts> [archivo2.ts ...]   Transpila los bucles paralelizables a WGSL
  ts-to-wgsl [-o <dir>]                                   Usa el tsconfig.json local

Opciones:
  -o, --out <dir>   Directorio de salida de los .wgsl (por defecto: ./wgsl)
  -h, --help        Muestra esta ayuda
`.trim();

function ejecutarCLI(): void {
    const args = process.argv.slice(2);

    if (args.includes('--help') || args.includes('-h')) {
        console.log(AYUDA);
        process.exit(0);
    }

    let dirSalida = 'wgsl';
    const archivosEntrada: string[] = [];
    for (let i = 0; i < args.length; i++) {
        const arg = args[i]!;
        if (arg === '-o' || arg === '--out') {
            const valor = args[++i];
            if (!valor) {
                console.error('[Error] Falta el directorio después de -o/--out');
                process.exit(1);
            }
            dirSalida = valor;
            continue;
        }
        const rutaAbsoluta = resolve(process.cwd(), arg);
        if (!existsSync(rutaAbsoluta)) {
            console.error(`[Error] No se encontró el archivo: ${arg}`);
            process.exit(1);
        }
        archivosEntrada.push(rutaAbsoluta);
    }

    try {
        const contexto = new TranspiladorTsaWGSL().transpilar(archivosEntrada);

        if (contexto.diagnosticos.length > 0) {
            console.warn('=== DIAGNÓSTICOS ===');
            console.dir(contexto.diagnosticos, { depth: null });
        }

        console.log(`Bucles candidatos: ${contexto.buclesCandidatos.length} | aptos para GPU: ${contexto.buclesAptos.length}`);

        if (contexto.shaders.length > 0) {
            const destino = resolve(process.cwd(), dirSalida);
            mkdirSync(destino, { recursive: true });
            for (const s of contexto.shaders) {
                const ruta = join(destino, `${s.nombre}.wgsl`);
                writeFileSync(ruta, s.wgsl + '\n');
                console.log(`  -> ${ruta}`);
            }
        }
    } catch (error) {
        console.error('[Error Fatal]:', error instanceof Error ? error.message : error);
        process.exit(1);
    }
}

ejecutarCLI();
