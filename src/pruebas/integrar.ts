// Avanza un paso de tiempo (Euler semi-implícito): v += a·dt y luego x += v·dt. Cada cuerpo solo toca su propio índice.
export function integrarCuerpos(x: number[], y: number[], vx: number[], vy: number[], ax: number[], ay: number[], n: number, dt: number) {
    for (let i = 0; i < n; i++) {
        vx[i] = vx[i]! + ax[i]! * dt;
        vy[i] = vy[i]! + ay[i]! * dt;
        x[i] = x[i]! + vx[i]! * dt;
        y[i] = y[i]! + vy[i]! * dt;
    }
}
