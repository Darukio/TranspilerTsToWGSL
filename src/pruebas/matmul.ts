export function multiplicarMatricesReal(A: number[], B: number[], C: number[], N: number) {
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            // Acumulador local por hilo: cada C[i][j] es el producto punto de una fila de A y una columna de B
            let suma = 0;
            for (let k = 0; k < N; k++) {
                suma += A[i * N + k]! * B[k * N + j]!;
            }
            C[i * N + j] = suma;
        }
    }
}
