export function sumar(a: number[], n: number) {
    let total = 0;
    for (let i = 0; i < n; i++) {
        total += a[i]!;
    }
    return total;
}
