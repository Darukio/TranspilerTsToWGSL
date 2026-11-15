import { crearDispositivo, describirDispositivo } from '../runtime/dispositivo.js';
import { crearServidor } from './servidor.js';

const indice = process.argv.findIndex((a) => a === '--puerto' || a === '-p');
const puerto = indice >= 0 ? Number(process.argv[indice + 1]) : 3000;
if (!Number.isInteger(puerto) || puerto < 0 || puerto > 65535) {
    console.error('Uso: node dist/server/main.js [--puerto <0-65535>]');
    process.exit(1);
}

try {
    const dispositivo = await crearDispositivo();
    const servidor = crearServidor(dispositivo);
    // Solo la máquina local: la página puede lanzar trabajo en la GPU. Dentro de un contenedor hay que
    // escuchar en 0.0.0.0 (BENCHMARK_HOST); ahí el puerto se publica solo en el loopback del anfitrión.
    const host = process.env.BENCHMARK_HOST ?? '127.0.0.1';
    servidor.listen(puerto, host, () => {
        console.log(describirDispositivo(dispositivo));
        console.log(`Benchmark en http://localhost:${(servidor.address() as { port: number }).port}`);
    });
    process.on('SIGINT', () => {
        servidor.close();
        dispositivo.destruir();
        process.exit(0);
    });
} catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
}
