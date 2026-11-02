import * as ts from 'typescript';
import type { IGestorAST } from '../core/IGestorAST.js';
import type { NodoBucle, Sentencia, Expr, Diagnostico, Ubicacion } from '../core/Modelo.js';

export class TSCompilerFacade implements IGestorAST {
    private program: ts.Program | undefined;
    private contador = 0;
    private registroNodos = new Map<string, ts.ForStatement>();

    constructor(programExistente?: ts.Program) {
        this.program = programExistente;
    }

    cargar(archivos: string[]): void {
        this.registroNodos.clear();
        this.contador = 0;

        // Si no pasan archivos explícitos, busca el tsconfig.json del proyecto actual
        if (archivos.length === 0) {
            const configPath = ts.findConfigFile(process.cwd(), ts.sys.fileExists, 'tsconfig.json');
            if (!configPath) {
                throw new Error('No se especificaron archivos ni se encontró un tsconfig.json.');
            }
            const configFile = ts.readConfigFile(configPath, ts.sys.readFile);
            const parsed = ts.parseJsonConfigFileContent(configFile.config, ts.sys, process.cwd());
            this.program = ts.createProgram(parsed.fileNames, parsed.options);
            return;
        }

        if (!this.program) {
            this.program = ts.createProgram(archivos, {
                target: ts.ScriptTarget.ESNext,
                strict: true,
            });
        }
    }

    obtenerTipoVariable(bucleId: string, nombreVar: string): string | undefined {
        if (!this.program) return undefined;
        const nodo = this.registroNodos.get(bucleId);
        if (!nodo) return undefined;
        const checker = this.program.getTypeChecker();
        // Más adelante aquí buscas el identificador dentro de 'nodo' y retornas checker.typeToString(...)
        return checker ? 'number' : undefined;
    }

    obtenerBuclesCandidatos(): NodoBucle[] {
        if (!this.program) return [];
        const resultado: NodoBucle[] = [];

        for (const sf of this.program.getSourceFiles()) {
            if (sf.isDeclarationFile || this.program.isSourceFileFromExternalLibrary(sf)) {
                continue;
            }

            const visitar = (n: ts.Node): void => {
                if (ts.isForStatement(n)) {
                    const b = this.convertirBucle(n, sf);
                    if (b) {
                        resultado.push(b);
                        // IMPORTANTE: Retornamos aquí porque this.convertirBucle ya
                        // procesa recursivamente los bucles hijos dentro de 'cuerpo'.
                        // Así evitamos duplicar bucles anidados como candidatos sueltos.
                        return;
                    }
                }
                ts.forEachChild(n, visitar);
            };
            visitar(sf);
        }
        return resultado;
    }

    obtenerDiagnosticos(): Diagnostico[] {
        if (!this.program) return [];
        return ts.getPreEmitDiagnostics(this.program).map(d => {
            const diag: Diagnostico = {
                fase: 'frontend',
                mensaje: ts.flattenDiagnosticMessageText(d.messageText, '\n'),
            };
            if (d.file && d.start !== undefined) {
                const { line, character } = d.file.getLineAndCharacterOfPosition(d.start);
                diag.ubicacion = { archivo: d.file.fileName, linea: line + 1, columna: character + 1 };
            }
            return diag;
        });
    }

    // ---- traducción ts.* -> modelo propio ----

    private convertirBucle(n: ts.ForStatement, sf: ts.SourceFile): NodoBucle | undefined {
        const init = n.initializer;
        if (!init || !ts.isVariableDeclarationList(init) || init.declarations.length !== 1) return undefined;
        const decl = init.declarations[0];
        if (!decl || !ts.isIdentifier(decl.name) || !decl.initializer) return undefined;

        const nombreVar = decl.name.text;

        // Validar condición: debe ser < o <= y el lado izquierdo DEBE ser la variable iteradora
        const cond = n.condition;
        if (!cond || !ts.isBinaryExpression(cond)) return undefined;
        if (!ts.isIdentifier(cond.left) || cond.left.text !== nombreVar) return undefined;

        const op = cond.operatorToken.kind;
        if (op !== ts.SyntaxKind.LessThanToken && op !== ts.SyntaxKind.LessThanEqualsToken) return undefined;

        // Validar paso asegurando que modifique a la variable iteradora
        const paso = this.convertirPaso(n.incrementor, nombreVar);
        if (!paso) return undefined;

        const id = `bucle_${this.contador++}`;
        this.registroNodos.set(id, n);

        return {
            id,
            ubicacion: this.ubicar(n, sf),
            variable: nombreVar,
            inicio: this.expr(decl.initializer, sf),
            limite: this.expr(cond.right, sf),
            inclusivo: op === ts.SyntaxKind.LessThanEqualsToken,
            paso,
            cuerpo: this.cuerpo(n.statement, sf),
        };
    }

    private convertirPaso(inc: ts.Expression | undefined, nombreVar: string): Expr | undefined {
        if (!inc) return undefined;
        // Caso i++ o ++i
        if (
            (ts.isPostfixUnaryExpression(inc) || ts.isPrefixUnaryExpression(inc)) &&
            inc.operator === ts.SyntaxKind.PlusPlusToken &&
            ts.isIdentifier(inc.operand) &&
            inc.operand.text === nombreVar
        ) {
            return { tipo: 'num', valor: 1 };
        }
        // Caso i += expr
        if (
            ts.isBinaryExpression(inc) &&
            inc.operatorToken.kind === ts.SyntaxKind.PlusEqualsToken &&
            ts.isIdentifier(inc.left) &&
            inc.left.text === nombreVar
        ) {
            return this.expr(inc.right, inc.getSourceFile());
        }
        return undefined;
    }

    private cuerpo(s: ts.Statement, sf: ts.SourceFile): Sentencia[] {
        const lista = ts.isBlock(s) ? s.statements : [s];
        return lista.map(x => this.sentencia(x, sf));
    }

    private sentencia(s: ts.Statement, sf: ts.SourceFile): Sentencia {
        if (ts.isForStatement(s)) {
            const b = this.convertirBucle(s, sf);
            if (b) return { tipo: 'bucle', bucle: b };
        }
        if (
            ts.isExpressionStatement(s) &&
            ts.isBinaryExpression(s.expression) &&
            s.expression.operatorToken.kind === ts.SyntaxKind.EqualsToken
        ) {
            return {
                tipo: 'asignacion',
                destino: this.expr(s.expression.left, sf),
                valor: this.expr(s.expression.right, sf),
            };
        }
        return { tipo: 'desconocida', texto: s.getText(sf) };
    }

    private expr(e: ts.Expression, sf: ts.SourceFile): Expr {
        if (ts.isParenthesizedExpression(e) || ts.isNonNullExpression(e)) return this.expr(e.expression, sf);
        if (ts.isNumericLiteral(e)) return { tipo: 'num', valor: Number(e.text) };

        // Soporte para literales negativos (ej. -5)
        if (
            ts.isPrefixUnaryExpression(e) &&
            e.operator === ts.SyntaxKind.MinusToken &&
            ts.isNumericLiteral(e.operand)
        ) {
            return { tipo: 'num', valor: -Number(e.operand.text) };
        }

        if (ts.isIdentifier(e)) return { tipo: 'var', nombre: e.text };

        if (ts.isBinaryExpression(e)) {
            const ops: Partial<Record<ts.SyntaxKind, '+' | '-' | '*' | '/'>> = {
                [ts.SyntaxKind.PlusToken]: '+',
                [ts.SyntaxKind.MinusToken]: '-',
                [ts.SyntaxKind.AsteriskToken]: '*',
                [ts.SyntaxKind.SlashToken]: '/',
            };
            const op = ops[e.operatorToken.kind];
            if (op) return { tipo: 'bin', op, izq: this.expr(e.left, sf), der: this.expr(e.right, sf) };
        }

        if (ts.isElementAccessExpression(e)) {
            const indices: Expr[] = [];
            let actual: ts.Expression = e;
            while (ts.isElementAccessExpression(actual)) {
                indices.unshift(this.expr(actual.argumentExpression, sf));
                actual = actual.expression;
            }
            if (ts.isIdentifier(actual)) return { tipo: 'acceso', arreglo: actual.text, indices };
        }

        // Soporte tanto para fn(x) como para Math.sqrt(x) (PropertyAccessExpression)
        if (ts.isCallExpression(e)) {
            if (ts.isIdentifier(e.expression)) {
                return {
                    tipo: 'llamada',
                    funcion: e.expression.text,
                    args: e.arguments.map(a => this.expr(a, sf)),
                };
            }
            if (ts.isPropertyAccessExpression(e.expression)) {
                return {
                    tipo: 'llamada',
                    funcion: e.expression.getText(sf), // ej. "Math.sqrt"
                    args: e.arguments.map(a => this.expr(a, sf)),
                };
            }
        }

        return { tipo: 'desconocida', texto: e.getText(sf) };
    }

    private ubicar(n: ts.Node, sf: ts.SourceFile): Ubicacion {
        const { line, character } = sf.getLineAndCharacterOfPosition(n.getStart(sf));
        return { archivo: sf.fileName, linea: line + 1, columna: character + 1 };
    }
}