# Ejecutar el prototipo con Docker (Linux + GPU NVIDIA)

El contenedor transpila TypeScript a WGSL y ejecuta ese WGSL en la GPU real del anfitrión.

## Requisitos (una sola vez)
1. Driver NVIDIA instalado en el anfitrión (`nvidia-smi` debe funcionar).
2. Docker Engine con **NVIDIA Container Toolkit** (`nvidia-ctk`), configurado como runtime de Docker.
   Los pasos 1 a 4 de abajo lo comprueban, lo instalan y lo configuran.

## Paso 1: ¿ya tengo el toolkit?
Ejecutar estos comandos; no modifican nada:
```bash
nvidia-smi                          # driver de NVIDIA: debe mostrar una tabla con tu GPU
command -v nvidia-ctk               # toolkit: imprime una ruta si está instalado, nada si no
dpkg -l | grep -i nvidia-container  # paquetes del toolkit (Debian/Ubuntu)
docker info | grep -i runtimes      # el runtime de Docker: debe incluir "nvidia"
```
- Si `nvidia-smi` falla: primero hay que instalar el driver de NVIDIA (el toolkit no lo trae).
- Si `command -v nvidia-ctk` no imprime nada: ir al paso 2.
- Si el toolkit está instalado pero `docker info` no menciona `nvidia`: saltar al paso 3 (falta configurar Docker).

## Paso 2: instalar el toolkit (Debian/Ubuntu, con apt)
Comandos tomados de la guía oficial de NVIDIA
(<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>):
```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
Para otras distribuciones (Fedora, RHEL, openSUSE) usar la guía oficial del enlace.

## Paso 3: refrescar Docker para que use el toolkit
Instalar el toolkit no basta: hay que decirle a Docker que lo use y reiniciarlo.
```bash
sudo nvidia-ctk runtime configure --runtime=docker   # agrega el runtime "nvidia" a /etc/docker/daemon.json
sudo systemctl restart docker                        # reinicia Docker para que lea el cambio
```

## Paso 4: comprobar que Docker ya ve la GPU
```bash
docker info | grep -i runtimes                              # ahora debe incluir "nvidia"
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```
El segundo comando debe imprimir la misma tabla de `nvidia-smi` que en el paso 1, pero **desde dentro de un
contenedor**. Si lo hace, Docker ya puede darle la GPU a cualquier contenedor.

Aviso de NVIDIA: en sistemas con el controlador de cgroups de systemd, un `sudo systemctl daemon-reload` puede hacer que
los contenedores pierdan el acceso a la GPU. Si pasa, reiniciar el contenedor (o Docker).

## Paso 5: uso
```bash
docker compose up --build
```
Abrir <http://localhost:3000>. El puerto se publica solo en el loopback del anfitrión.

## Comprobar que usa la GPU real
```bash
docker compose run --rm transpilador node dist/runtime/info.js
```
Debe decir `hardware`. Si dice `SOFTWARE`, el contenedor no recibió la GPU (revisar el toolkit) y las
mediciones de rendimiento **no valen**; la corrección sí se puede verificar.

### Ver otra opción para probar el transpilador
Si no tienes el NVIDIA Container Toolkit, hay una vía sin Docker: leer el [`README.md`](../README.md).

## Notas
- Para una GPU AMD/Intel en Linux, sustituir el bloque `deploy` del compose por `devices: ["/dev/dri"]`.
- Variable `BENCHMARK_HOST` (por defecto `127.0.0.1`; la imagen la fija en `0.0.0.0`).
