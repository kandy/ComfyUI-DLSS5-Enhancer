#!/usr/bin/env bash
# Build the native in-process DLSS Python binding (libdlss_native.so).
#
# The binding drives the REAL NVIDIA DLSS SuperSampling neural engine
# (NVSDK_NGX_CUDA_* via libnvsdk_ngx.a + the driver libnvidia-ngx.so.1),
# with a Lanczos-4 fallback if NGX is unavailable.
#
# Auto-detection (each overridable via env):
#   host arch     uname -m -> Linux_x86_64 / Linux_aarch64
#   GPU arch      nvidia-smi compute_cap -> sm_86, sm_89, sm_120, ...
#   CUDA toolkit  nvcc on PATH or $CUDA_HOME/bin/nvcc, default /usr/local/cuda
#   DLSS SDK      bundled ../DLSS, then $DLSS_SDK_DIR
#   arch in SDK   lib/Linux_$HOST_ARCH
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # .../dlss/dlss_native
WORKSPACE="$(cd "$HERE/.." && pwd)"            # .../dlss
cd "$HERE"
mkdir -p build

# ---- host arch ---------------------------------------------------------------
HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    x86_64)        HOST_ARCH=x86_64 ;;
    aarch64|arm64) HOST_ARCH=aarch64 ;;
    *) echo "error: unsupported host arch: $HOST_ARCH" >&2; exit 1 ;;
esac

# ---- GPU arch ----------------------------------------------------------------
# ARCH env wins; else the first GPU's compute capability via nvidia-smi;
# else a per-host default (sm_120 for aarch64 DGX Spark, sm_86 for x86_64).
if [[ -z "${ARCH:-}" ]]; then
    CC=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')" || true
    fi
    if [[ "$CC" =~ ^[0-9]+\.[0-9]+$ ]]; then
        ARCH="sm_$(echo "$CC" | tr -d '.')"
    elif [[ "$HOST_ARCH" == "aarch64" ]]; then
        ARCH=sm_120
    else
        ARCH=sm_86
    fi
fi

# ---- CUDA toolkit ------------------------------------------------------------
# NVCC: nvcc on PATH, or $CUDA_HOME/bin/nvcc. CUDA_HOME defaults to the
# directory containing nvcc on PATH, or /usr/local/cuda.
if [[ -z "${NVCC:-}" ]]; then
    NVCC="$(command -v nvcc 2>/dev/null || true)"
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -n "$NVCC" && -x "$NVCC" ]]; then
        CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
    else
        CUDA_HOME=/usr/local/cuda
    fi
fi
if [[ ! -x "${NVCC:-}" ]]; then
    NVCC="$CUDA_HOME/bin/nvcc"
fi
CUDA_INCLUDE="${CUDA_INCLUDE:-$CUDA_HOME/include}"
CUDA_LIB="${CUDA_LIB:-$CUDA_HOME/lib64}"

# ---- DLSS SDK tree -----------------------------------------------------------
# Check out the NVIDIA DLSS SDK into the pack's DLSS directory unless a tree
# is already there: DLSS_SDK_DIR points at an existing tree; otherwise the
# SDK is cloned from DLSS_SDK_REPO (default: the official NVIDIA/DLSS repo).
SDK="${DLSS_SDK_DIR:-}"
if [[ -z "$SDK" ]]; then
    if [[ -d "$WORKSPACE/DLSS/include" ]]; then
        SDK="$WORKSPACE/DLSS"
    else
        SDK_REPO="${DLSS_SDK_REPO:-https://github.com/NVIDIA/DLSS}"
        echo "cloning DLSS SDK: $SDK_REPO -> $WORKSPACE/DLSS"
        git clone --depth 1 "$SDK_REPO" "$WORKSPACE/DLSS"
        SDK="$WORKSPACE/DLSS"
    fi
fi
SDK_LIB="$SDK/lib/Linux_$HOST_ARCH"

echo "building for host=$HOST_ARCH arch=$ARCH cuda=$CUDA_HOME sdk=$SDK_LIB"

if [[ -z "$SDK" ]]; then
    echo "error: DLSS SDK not found (searched bundled ../DLSS; set DLSS_SDK_DIR)." >&2
    exit 1
fi
if [[ ! -f "$SDK_LIB/libnvsdk_ngx.a" ]]; then
    echo "error: DLSS SDK has no static wrapper for this host arch ($HOST_ARCH)." >&2
    echo "  expected: $SDK_LIB/libnvsdk_ngx.a" >&2
    echo "  set DLSS_SDK_DIR to a tree providing include/ and lib/Linux_$HOST_ARCH/libnvsdk_ngx.a" >&2
    exit 1
fi

"$NVCC" -O3 -arch="$ARCH" -shared -o build/libdlss_native.so \
    src/libdlss_native.cu \
    -I "$HERE/include" -I "$HERE/include/nvofapi" \
    -I "$SDK/include" -I "$CUDA_INCLUDE" \
    -L "$SDK_LIB" -L "$CUDA_LIB" \
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
