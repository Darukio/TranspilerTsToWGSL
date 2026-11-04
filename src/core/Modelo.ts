export interface Ubicacion {
    archivo: string;
    linea: number;
    columna: number;
}

export type Expr =
    | { tipo: 'num'; valor: number }
    | { tipo: 'var'; nombre: string }
    | { tipo: 'bin'; op: '+' | '-' | '*' | '/'; izq: Expr; der: Expr }
    | { tipo: 'acceso'; arreglo: string; indices: Expr[] }   // a[i][j] -> indices [i, j]
    | { tipo: 'llamada'; funcion: string; args: Expr[] }
    | { tipo: 'desconocida'; texto: string };

export type Sentencia =
    | { tipo: 'asignacion'; destino: Expr; valor: Expr }
    | { tipo: 'declaracion'; nombre: string; valor: Expr }   // let suma = 0;
    | { tipo: 'bucle'; bucle: NodoBucle }
    | { tipo: 'desconocida'; texto: string };

export interface NodoBucle {
    id: string;
    ubicacion: Ubicacion;
    variable: string;
    inicio: Expr;
    limite: Expr;
    inclusivo: boolean;   // i <= n  vs  i < n
    paso: Expr;
    cuerpo: Sentencia[];
}

export interface Diagnostico {
    fase: string;
    mensaje: string;
    ubicacion?: Ubicacion;
}