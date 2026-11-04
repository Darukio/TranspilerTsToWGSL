#!/usr/bin/env bash
# Uso:  source scripts/wsl-gpu.sh
# Hace que Dawn (Node en WSL) use la GPU real vía Dozen (Vulkan sobre Direct3D12) en lugar de llvmpipe.
# Requiere el Mesa con dzn compilado en ~/.local/mesa-dzn (ver docs/GPU-EN-WSL.md).
DZN="$HOME/.local/mesa-dzn"
export VK_DRIVER_FILES="$DZN/share/vulkan/icd.d/dzn_icd.x86_64.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$DZN/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
