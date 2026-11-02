export function multiplicarMatrices(A: number[], B: number[], C: number[], N: number) {
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            // Acceso afín bidimensional indexado linealmente
            C[i * N + j] = A[i * N + j]! + B[i * N + j]!;
        }
    }
}