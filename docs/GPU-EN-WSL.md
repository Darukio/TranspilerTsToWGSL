# GPU real en WSL (Dozen)

> **Para medir rendimiento usar Windows nativo** (ver `docs/BENCHMARK.md`): el camino WSL + Dozen agrega entre 10 y 100 veces de latencia en sincronización y lectura, y hace más lento el propio cómputo. Esta guía sirve para tests de corrección dentro de WSL.

Dawn en Linux solo habla Vulkan, y WSL no expone `/dev/dri`. Sin configuración, Dawn solo ve
`llvmpipe` (la CPU emulando una GPU): sirve para probar la corrección, **no** para medir rendimiento.

La salida es Dozen (`dzn`): Vulkan sobre Direct3D12, que el Mesa de Ubuntu no incluye.

## Qué se hizo (una sola vez)
1. Dependencias de compilación con `apt` (meson, ninja, g++, directx-headers-dev, spirv-tools, ...).
2. Mesa 26.0.3 (`archive.mesa3d.org`) compilado solo con `-Dvulkan-drivers=microsoft-experimental`,
   sin LLVM ni OpenGL, instalado en `~/.local/mesa-dzn` (no toca el sistema).
3. **Parche local** en `src/microsoft/vulkan/dzn_device.c`: `fullDrawIndexUint32 = true`.
   Dawn rechaza cualquier adaptador sin esa característica, aunque solo se use cómputo.
   El parche solo declara la característica; no se usan dibujos con índices de 32 bits.
   `dzn` no es una implementación Vulkan conforme: usar para pruebas y mediciones de este proyecto.

## Uso
```bash
source scripts/wsl-gpu.sh
pnpm gpu-info     # debe decir "Dozen ... - microsoft-direct3d12-amd-radeon-..." y "hardware"
```
Sin el `source`, `gpu-info` dirá `SOFTWARE`.

## Benchmark en el navegador
```bash
source scripts/wsl-gpu.sh
pnpm servidor            # http://localhost:3000 (solo 127.0.0.1)
```
La página lanza el benchmark de cada caso (calentamiento + repeticiones, mediana) y grafica CPU, GPU total
(subida + cómputo + lectura) y GPU solo cómputo. Si el adaptador es de software, muestra un aviso: esa
comparación no vale.

Notas de medición:
- El cómputo puro se mide en corridas aparte con `timestamp-query`: leer ese cronómetro cuesta una lectura
  extra y contaminaría el tiempo total.
- Dawn redondea los timestamps a 65 µs. `crearDispositivo` desactiva esa protección
  (`disable-dawn-features=timestamp_quantization`) porque aquí todo corre en local.
- La línea base de CPU es el mismo código TypeScript compilado por `tsc`, sobre `Float64Array`.
