#!/bin/sh

vulkan_lib_dir=/run/current-system/sw/lib

if [ -e "$vulkan_lib_dir/libvulkan.so.1" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$vulkan_lib_dir:"*) ;;
        *) LD_LIBRARY_PATH="$vulkan_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
    export LD_LIBRARY_PATH
fi
