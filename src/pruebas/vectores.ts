export function procesarVectores(vectorA: number[], vectorB: number[], salida: number[], n: number) {
    for (let i = 0; i < n; i++) {
        // Ninguna iteración depende del resultado de otra (paralelismo puro en GPU)
        salida[i] = (vectorA[i]! * 3.14) - vectorB[i]!;
    }
}