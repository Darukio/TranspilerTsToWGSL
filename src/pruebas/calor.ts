// Difusión de calor en una placa N×N: un paso de la ecuación del calor. Lee `actual`, escribe `siguiente` (doble búfer).
// Solo se actualiza el interior; los bordes quedan fijos.
export function difundirCalor(actual: number[], siguiente: number[], N: number, alfa: number) {
    for (let i = 0; i < N - 2; i++) {
        for (let j = 0; j < N - 2; j++) {
            const c = (i + 1) * N + (j + 1);
            siguiente[c] = actual[c]! + alfa * (actual[c - N]! + actual[c + N]! + actual[c - 1]! + actual[c + 1]! - 4 * actual[c]!);
        }
    }
}
