# GPU real en NixOS (Vulkan)

En NixOS el loader de Vulkan (`libvulkan.so.1`) está en `/run/current-system/sw/lib`, fuera de la ruta que Node usa para `dlopen`. Sin configurarlo, Dawn (paquete `webgpu`) falla con:

```text
Warning: Couldn't load Vulkan: libvulkan.so.1: cannot open shared object file: No such file or directory
No hay adaptador WebGPU disponible (¿drivers Vulkan/Metal/D3D12 instalados?).
```

El proyecto ya lo resuelve solo: los scripts `test`, `gpu-info` y `servidor` cargan `scripts/gpu-env.sh`, que agrega esa ruta a `LD_LIBRARY_PATH` únicamente si existe. En macOS, Windows u otra distro no hace nada.

## Uso

```bash
pnpm gpu-info
pnpm test
pnpm servidor
```

Para un script propio que use WebGPU:

```bash
. ./scripts/gpu-env.sh
node mi-script.js
```

## Verificación

`pnpm gpu-info` debe mostrar la GPU real, por ejemplo:

```text
radv: Mesa ... - amd-radeon-graphics-radv-... | hardware | timestamp-query: sí
```

Si aparece `SOFTWARE` o el error de Vulkan, revisar que exista `/run/current-system/sw/lib/libvulkan.so.1`.
