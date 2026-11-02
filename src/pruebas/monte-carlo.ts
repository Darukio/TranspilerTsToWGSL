export function simularMonteCarlo(precios: number[], resultados: number[], n: number) {
    for (let i = 0; i < n; i++) {
        // Operaciones aritméticas y funciones matemáticas puras aptas para WGSL
        resultados[i] = Math.sqrt(precios[i]! * 2.5) + Math.sin(precios[i]!);
    }
}