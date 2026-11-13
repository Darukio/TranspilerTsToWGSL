# TranspilerTsToWGSL

Transpilador de TypeScript a WGSL: detecta los bucles `for` paralelizables, los convierte en *compute shaders*
y los ejecuta en la GPU con WebGPU. Incluye un servidor local con el benchmark CPU contra GPU y las simulaciones.

## Probarlo sin Docker (Linux + GPU NVIDIA)

Es el camino más corto: Node habla directo con el driver NVIDIA del equipo, sin contenedor ni NVIDIA Container
Toolkit. Si prefieres Docker, mira [`docs/DOCKER.md`](docs/DOCKER.md).

### Requisitos
- Driver propietario de NVIDIA instalado (`nvidia-smi` debe funcionar). Incluye el driver Vulkan que usa WebGPU.
- Node.js 24 (probado con 24.18) y `corepack` (viene con Node).
- La librería del cargador Vulkan: `sudo apt install libvulkan1` (Debian/Ubuntu).

### Pasos
```bash
git clone https://github.com/Darukio/TranspilerTsToWGSL.git
cd TranspilerTsToWGSL
corepack enable
pnpm install
```

1. Comprobar que ve la GPU real:
   ```bash
   pnpm gpu-info
   ```
   Debe terminar en `hardware`. Si dice `SOFTWARE`, WebGPU no encontró el driver Vulkan de NVIDIA y las
   mediciones de rendimiento **no valen**.

2. Levantar el benchmark y las simulaciones:
   ```bash
   pnpm servidor
   ```
   Abrir <http://localhost:3000> (benchmark) y <http://localhost:3000/simulacion> (simulaciones).
   El servidor solo atiende a `localhost`.

3. Transpilar un archivo a WGSL (genera los `.wgsl` en `./wgsl`):
   ```bash
   pnpm ts-to-wgsl -o wgsl src/pruebas/vectores.ts
   ```

4. Ejecutar los tests (los de GPU se omiten si no hay adaptador):
   ```bash
   pnpm test
   ```

> Estado de la verificación: esta vía se probó con GPU AMD en WSL2 (Dozen) y el contenedor con GPU por software.
> **No se ha probado todavía con una GPU NVIDIA real.**
