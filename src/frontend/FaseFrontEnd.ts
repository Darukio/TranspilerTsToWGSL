import type { IFase } from "../core/IFase.js";
import type { ContextoPipeline } from "../core/ContextoPipeline.js";
import type { IGestorAST } from "../core/IGestorAST.js";
import type { NodoBucle } from "../core/Modelo.js";
import type { IReglaSemantica } from "./IReglaSemantica.js";
import { ValidacionPureza, ValidacionTipos } from "./ReglasSemanticas.js";

export class FaseFrontEnd implements IFase {
    constructor(
        private readonly gestor: IGestorAST,
        private readonly reglas: IReglaSemantica[] = [new ValidacionTipos(), new ValidacionPureza()],
    ) { }

    ejecutar(ctx: ContextoPipeline): ContextoPipeline {
        this.gestor.cargar(ctx.archivosEntrada);
        const candidatos = this.gestor.obtenerBuclesCandidatos();
        ctx.buclesCandidatos = this.filtrarPorReglas(candidatos, ctx);
        ctx.diagnosticos.push(...this.gestor.obtenerDiagnosticos());
        return ctx;
    }

    private filtrarPorReglas(candidatos: NodoBucle[], ctx: ContextoPipeline): NodoBucle[] {
        const aptos: NodoBucle[] = [];
        for (const bucle of candidatos) {
            const error = this.primerError(bucle);
            if (!error) {
                aptos.push(bucle);
                continue;
            }
            ctx.diagnosticos.push({
                fase: 'frontend',
                mensaje: `${bucle.id} (${error.regla}): ${error.mensaje}`,
                ubicacion: bucle.ubicacion,
            });
        }
        return aptos;
    }

    private primerError(bucle: NodoBucle): { regla: string; mensaje: string } | undefined {
        for (const regla of this.reglas) {
            const resultado = regla.validar(bucle);
            if (!resultado.ok) {
                return { regla: regla.nombre, mensaje: resultado.mensaje };
            }
        }
        return undefined;
    }
}