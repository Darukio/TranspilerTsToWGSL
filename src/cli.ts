#!/usr/bin/env node
// src/cli.ts
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { TranspiladorTsaWGSL } from './core/TranspiladorTSaWGSL.js';

function ejecutarCLI(): void {
    const args = process.argv.slice(2);

    if (args.includes('--help') || args.includes('-h')) {
        console.log(`
Uso:
  ts-to-wgsl <archivo1.ts> [archivo2.ts ...]   Analiza los archivos indicados
  ts-to-wgsl                                   Analiza el proyecto según tsconfig.json local
        `.trim());
        process.exit(0);
    }

    // Convertimos las rutas pasadas por consola a rutas absolutas y verificamos que existan
    const archivosEntrada: string[] = [];
    for (const arg of args) {
        const rutaAbsoluta = resolve(process.cwd(), arg);
        if (!existsSync(rutaAbsoluta)) {
            console.error(`[Error] No se encontró el archivo: ${arg}`);
            process.exit(1);
        }
        archivosEntrada.push(rutaAbsoluta);
    }

    try {
        const transpilador = new TranspiladorTsaWGSL();
        const contexto = transpilador.transpilar(archivosEntrada);

        if (contexto.diagnosticos.length > 0) {
            console.warn('=== DIAGNÓSTICOS ===');
            console.dir(contexto.diagnosticos, { depth: null });
        }

        console.log(`=== BUCLES CANDIDATOS DETECTADOS (${contexto.buclesCandidatos.length}) ===`);
        console.dir(contexto.buclesCandidatos, { depth: null });

    } catch (error) {
        console.error('[Error Fatal]:', error instanceof Error ? error.message : error);
        process.exit(1);
    }
}

ejecutarCLI();