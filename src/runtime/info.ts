import { crearDispositivo, describirDispositivo } from './dispositivo.js';

try {
    const dispositivo = await crearDispositivo();
    console.log(describirDispositivo(dispositivo));
    dispositivo.destruir();
} catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
}
