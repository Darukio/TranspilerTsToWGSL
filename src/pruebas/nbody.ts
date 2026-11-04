// Atracción gravitatoria entre todos los pares de cuerpos (2D). Cada cuerpo i suma la fuerza de los n cuerpos j: O(n²).
export function calcularFuerzas(x: number[], y: number[], masa: number[], ax: number[], ay: number[], n: number, suavizado: number) {
    for (let i = 0; i < n; i++) {
        let fx = 0;
        let fy = 0;
        for (let j = 0; j < n; j++) {
            const dx = x[j]! - x[i]!;
            const dy = y[j]! - y[i]!;
            const d2 = dx * dx + dy * dy + suavizado;
            const inv = 1 / (d2 * Math.sqrt(d2));
            fx += masa[j]! * dx * inv;
            fy += masa[j]! * dy * inv;
        }
        ax[i] = fx;
        ay[i] = fy;
    }
}
