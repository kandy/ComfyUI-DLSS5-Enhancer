#!/usr/bin/env bash
# Build the native in-process DLSS Python binding (libdlss_native.so).
#
# The binding drives the REAL NVIDIA DLSS SuperSampling neural engine
# (NVSDK_NGX_CUDA_* via DLSS/lib/Linux_aarch64/libnvsdk_ngx.a + the driver
# libnvidia-ngx.so.1), with a Lanczos-4 fallback if NGX is unavailable.
# Target: Linux aarch64 + NVIDIA GPU (validated on DGX Spark / GB10 sm_120).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # .../dlss/dlss_native
WORKSPACE="$(cd "$HERE/.." && pwd)"            # .../dlss
cd "$HERE"
mkdir -p build

ARCH="${ARCH:-sm_120}"
CUDA_INCLUDE="${CUDA_INCLUDE:-/usr/local/cuda/include}"
CUDA_LIB="${CUDA_LIB:-/usr/local/cuda/lib64}"

# Locate the DLSS SDK tree: bundled pack first, then the dev workspace, then
# an explicit override. Distribution packs carry it in ../DLSS.
SDK="${DLSS_SDK_DIR:-}"
if [[ -z "$SDK" ]]; then
    for cand in "$HERE/../DLSS" "$HERE/../ComfyUI-DLSS5-Enhancer/DLSS" "$WORKSPACE/DLSS"; do
        if [[ -d "$cand/include" ]]; then SDK="$(cd "$cand" && pwd)"; break; fi
    done
fi
if [[ -z "$SDK" || ! -f "$SDK/lib/Linux_aarch64/libnvsdk_ngx.a" ]]; then
    echo "error: DLSS SDK not found. Set DLSS_SDK_DIR to the folder containing include/ and lib/Linux_aarch64/." >&2
    exit 1
fi

nvcc -O3 -arch="$ARCH" -shared -o build/libdlss_native.so \
    src/libdlss_native.cu \
    -I "$HERE/include" -I "$HERE/include/nvofapi" \
    -I "$SDK/include" -I "$CUDA_INCLUDE" \
    -L "$SDK/lib/Linux_aarch64" -L "$CUDA_LIB" \
    -lnvsdk_ngx -ldl -lcuda -lcudart -Xcompiler -fPIC

echo "Built: $HERE/build/libdlss_native.so"

# smoke-test the binding right away
DLSS_NATIVE_HOME="$HERE" python3 -B - "$WORKSPACE" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])                # workspace, so `import dlss_native` works
import dlss_native as D
assert D.is_available(), "native binding not initialized"
print("binding is live on GPU: True")
PY
