# Entorno y herramientas

| Categoría | Nombre | Versión | Uso / Nota |
|---|---|---|---|
| Hardware | CPU AMD Ryzen AI 9 365 | 20 hilos | Líneas base CPU 1 hilo y N hilos |
| Hardware | GPU AMD Radeon 880M (integrada) | — | Memoria compartida con la CPU |
| Sistema operativo | Windows | 10.0.26200 (build 26200.9711) | Entorno de medición de rendimiento (Direct3D 12) |
| Sistema operativo | Ubuntu sobre WSL2 | 26.04 LTS | Pruebas de corrección |
| Sistema operativo | Kernel de WSL2 | 6.18.33.2-microsoft-standard-WSL2 | — |
| Ejecución | Node.js (Windows) | v24.20.0 | Ejecución del transpilador y del benchmark |
| Ejecución | Node.js (WSL) | v24.18.0 | Ejecución de la suite de pruebas |
| Gestor de paquetes | pnpm | ^11.13.1 | Declarado en `devEngines` |
| Compilador | TypeScript (`@typescript/typescript6`) | 6.0.2 | Compilación y API del frontend (`ts.createProgram`) |
| Compilador | `@typescript/native` | ^7.0.2 (rango; no instalado en WSL) | Declarado en `devDependencies` |
| Biblioteca | webgpu (Dawn para Node) | 0.6.2 | Implementación de WebGPU (puente CPU–GPU) |
| Biblioteca | zod | 4.6.7 | Contrato del IR (`KernelIRSchema`) |
| Tipos | @webgpu/types | ^0.1.74 (rango) | Tipos de la API WebGPU |
| Tipos | @types/node | ^26.6.5 (rango) | Tipos de Node.js |
| API gráfica (Windows) | Direct3D 12 | No verificada | Camino Node → Dawn → D3D12 → GPU |
| API gráfica (WSL) | Vulkan sobre Dozen (Mesa) | Mesa 26.0.3 con parche local | Solo corrección; agrega 10 a 100 veces de latencia |
| Emulador de GPU | llvmpipe | No verificada | Pruebas sin GPU real |
| Lenguaje de salida | WGSL | Según la implementación de Dawn | Compute shaders generados |
| Driver de GPU (Windows) | AMD Radeon | No verificada | Ver Administrador de dispositivos o `dxdiag` |
