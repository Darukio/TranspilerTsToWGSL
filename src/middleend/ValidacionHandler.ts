export interface ReductorValidable {
    configurarGrid(): void;
    analizarCuerpo(): void;
    verificarSinColisiones(): void;
}

export abstract class ValidacionHandler {
    private siguiente: ValidacionHandler | undefined;

    enlazar(siguiente: ValidacionHandler): ValidacionHandler {
        this.siguiente = siguiente;
        return siguiente;
    }

    ejecutar(reductor: ReductorValidable): void {
        this.validar(reductor);
        this.siguiente?.ejecutar(reductor);
    }

    protected abstract validar(reductor: ReductorValidable): void;
}

export class ScopeHandler extends ValidacionHandler {
    protected validar(reductor: ReductorValidable): void {
        reductor.configurarGrid();
    }
}

export class PoliedralHandler extends ValidacionHandler {
    protected validar(reductor: ReductorValidable): void {
        reductor.analizarCuerpo();
    }
}

export class BernsteinHandler extends ValidacionHandler {
    protected validar(reductor: ReductorValidable): void {
        reductor.verificarSinColisiones();
    }
}

export function crearCadenaValidacion(): ValidacionHandler {
    const inicio = new ScopeHandler();
    inicio
        .enlazar(new PoliedralHandler())
        .enlazar(new BernsteinHandler());
    return inicio;
}
