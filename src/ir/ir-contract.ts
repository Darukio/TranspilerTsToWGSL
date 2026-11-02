import { z } from "zod";

export const DataType = z.enum(["i32", "u32", "f32"]);
export type DataType = z.infer<typeof DataType>;

export const AccessMode = z.enum(["read", "write", "readwrite"]);
export type AccessMode = z.infer<typeof AccessMode>;

export type Expr =
  | { op: "lit"; type: DataType; value: number }
  | { op: "ref"; id: string }
  | { op: "read"; buffer: string; index: Expr }
  | { op: "bin"; operator: "+" | "-" | "*" | "/" | "<" | ">" | "=="; l: Expr; r: Expr }
  | { op: "unary"; operator: "-" | "!"; x: Expr }
  | { op: "cast"; targetType: DataType; x: Expr }
  | { op: "call"; func: string; args: Expr[] }
  | { op: "select"; cond: Expr; a: Expr; b: Expr };

export type Stmt =
  | { op: "var"; id: string; type: DataType; init: Expr }
  | { op: "set"; id: string; value: Expr }
  | { op: "store"; buffer: string; index: Expr; value: Expr }
  | { op: "for"; iter: string; from: Expr; to: Expr; body: Stmt[] };

export const ExprSchema: z.ZodType<Expr> = z.lazy(() =>
  z.discriminatedUnion("op", [
    z.object({ op: z.literal("lit"), type: DataType, value: z.number() }),
    z.object({ op: z.literal("ref"), id: z.string() }),
    z.object({ op: z.literal("read"), buffer: z.string(), index: ExprSchema }),
    z.object({
      op: z.literal("bin"),
      operator: z.enum(["+", "-", "*", "/", "<", ">", "=="]),
      l: ExprSchema,
      r: ExprSchema,
    }),
    z.object({ op: z.literal("unary"), operator: z.enum(["-", "!"]), x: ExprSchema }),
    z.object({ op: z.literal("cast"), targetType: DataType, x: ExprSchema }),
    z.object({ op: z.literal("call"), func: z.string(), args: z.array(ExprSchema) }),
    z.object({ op: z.literal("select"), cond: ExprSchema, a: ExprSchema, b: ExprSchema }),
  ])
);

export const StmtSchema: z.ZodType<Stmt> = z.lazy(() =>
  z.discriminatedUnion("op", [
    z.object({ op: z.literal("var"), id: z.string(), type: DataType, init: ExprSchema }),
    z.object({ op: z.literal("set"), id: z.string(), value: ExprSchema }),
    z.object({ op: z.literal("store"), buffer: z.string(), index: ExprSchema, value: ExprSchema }),
    z.object({
      op: z.literal("for"),
      iter: z.string(),
      from: ExprSchema,
      to: ExprSchema,
      body: z.array(StmtSchema),
    }),
  ])
);

export const KernelIRSchema = z.object({
  irVersion: z.literal(1),
  name: z.string(),
  params: z.array(z.object({ name: z.string(), type: DataType })),
  buffers: z.array(z.object({ name: z.string(), type: DataType, access: AccessMode })),
  grid: z.array(z.object({ iter: z.string(), size: ExprSchema })).min(1).max(3),
  body: z.array(StmtSchema),
});

export type KernelIR = z.infer<typeof KernelIRSchema>;
