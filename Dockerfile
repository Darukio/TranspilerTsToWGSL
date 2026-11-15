FROM node:24-trixie-slim

# libvulkan1: Dawn (WebGPU en Node) habla Vulkan. mesa-vulkan-drivers solo da una GPU por software
# como respaldo; con NVIDIA Container Toolkit el driver real de NVIDIA tiene prioridad.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# El toolkit solo inyecta el driver Vulkan si se piden las capacidades "graphics" y "compute".
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    BENCHMARK_HOST=0.0.0.0

WORKDIR /app
RUN corepack enable

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

COPY tsconfig.json ./
COPY src ./src
COPY web ./web
RUN pnpm build

EXPOSE 3000
CMD ["node", "dist/server/main.js"]
