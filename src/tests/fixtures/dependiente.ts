export function desplazar(a: number[], n: number) {
    for (let i = 0; i < n; i++) {
        a[i] = a[i + 1]! * 2;
    }
}
