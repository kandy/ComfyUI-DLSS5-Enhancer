// =============================================================================
// libdlss_native.cu  --  Native Linux CUDA DLSS image upscaler (Python binding).
//
//   Flat C ABI for ctypes (see dlss_native/_native.py), in-process.
//   The upscale path drives the REAL NVIDIA DLSS SuperSampling neural engine
//   (NVSDK_NGX_CUDA_* via libnvsdk_ngx.a + driver libnvidia-ngx.so.1),
//   following the proven recipe in dlss_sr_sample/:
//
//    * Init needs a current CUDA context first; app-data path must be writable.
//    * DLSS plugin discovery via FeatureCommonInfo.PathListInfo, resolved
//      from this .so's own location (dladdr) so installs anywhere work.
//    * CUDA resources are CUtexObject (sampled inputs) / CUsurfObject
//      (output), passed as POINTERS. The RGBA8 photography path is broken in
//      plugin 310.9.1 (white output), so color runs as RGBA16F linear with
//      sRGB decode/encode kernels around it.
//    * Full eval param set: Color/Output/Depth/MotionVectors, Jitter(0),
//      Sharpness(0), Reset, MVScale(1), RenderSubrectDims, PreExposure/
//      ExposureScale(1). Anything missing -> 0xBAD00005; raw device pointers
//      as resources -> host segfault inside the plugin.
//   On any NGX failure the call transparently falls back to the separable
//   Lanczos-4 resizer below, and dlss_last_path() reports which path ran.
//
//   Build (aarch64/Linux):
//     nvcc -O3 -arch=sm_120 -shared -o build/libdlss_native.so \
//         src/libdlss_native.cu \
//         -I ../../DLSS/include -L ../../DLSS/lib/Linux_aarch64 \
//         -lnvsdk_ngx -ldl -lcuda -lcudart -Xcompiler -fPIC
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_defs.h"

// Forward declaration (used by dladdr-based plugin path discovery below).
extern "C" int32_t dlss_init(int32_t app_id);

// ---- result codes (mirror NVSDK_NGX_Result_Success == 1) -------------------
#define DLSS_OK                 (int32_t)0x000001
#define DLSS_ERR_BAD_INPUT      (int32_t)0xBAD00005
#define DLSS_ERR_NOT_INIT       (int32_t)0xBAD00007
#define DLSS_ERR_CUDA           (int32_t)0xBAD00002
#define DLSS_ERR_OOM            (int32_t)0xBAD0000D

// NVSDK quality enum values (nvsdk_ngx_defs.h)
#define QUALITY_MAXPERF   0
#define QUALITY_BALANCED  1
#define QUALITY_MAXQUALITY 2
#define QUALITY_ULTRAPERF 3
#define QUALITY_DLAA      5

static const char *g_last_error = "no error";
static void set_err(const char *msg) { g_last_error = msg; }

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

// ---- global state ----------------------------------------------------------
struct NgxFeat {
    bool active = false;
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    int quality = -1, preset = 0;
    CUarray aC = nullptr, aO = nullptr, aM = nullptr, aD = nullptr;
    CUtexObject tC = 0, tM = 0, tD = 0;
    CUsurfObject sO = 0;
    void *d_in8 = nullptr;    // RGBA8 upload staging
    __half *d_in16 = nullptr; // RGBA16F linear
    void *d_mv = nullptr;     // RG16F motion staging
    __half *d_out16 = nullptr;// RGBA16F linear output
    void *d_out8 = nullptr;   // RGBA8 download staging
    NVSDK_NGX_Parameter *params = nullptr;
    NVSDK_NGX_Handle *feat = nullptr;
};

struct NativeState {
    cudaStream_t stream = nullptr;
    bool initialised = false;
    bool ngx_available = false;
    int last_path = 0; // 1 = neural DLSS, 0 = Lanczos fallback
    int32_t enh_intensity = 20, enh_tone = 10, enh_struct = 20;
    NgxFeat dlss;
};
static NativeState g;

// ---- sRGB <-> linear -------------------------------------------------------
__device__ __forceinline__ float dev_srgb_to_lin(float c) {
    return c <= 0.04045f ? c * (1.0f / 12.92f)
                         : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}
__device__ __forceinline__ float dev_lin_to_srgb(float c) {
    return c <= 0.0031308f ? c * 12.92f
                           : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

__global__ void k_rgba8_to_rgba16f(const uint8_t* __restrict__ src,
                                   __half* __restrict__ dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        dst[i * 4 + 0] = __float2half(dev_srgb_to_lin(src[i * 4 + 0] * (1.0f / 255.0f)));
        dst[i * 4 + 1] = __float2half(dev_srgb_to_lin(src[i * 4 + 1] * (1.0f / 255.0f)));
        dst[i * 4 + 2] = __float2half(dev_srgb_to_lin(src[i * 4 + 2] * (1.0f / 255.0f)));
        dst[i * 4 + 3] = __float2half(src[i * 4 + 3] * (1.0f / 255.0f));
    }
}

__global__ void k_rgba16f_to_rgba8(const __half* __restrict__ src,
                                   uint8_t* __restrict__ dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        float r = dev_lin_to_srgb(fmaxf(0.f, __half2float(src[i * 4 + 0])));
        float gg = dev_lin_to_srgb(fmaxf(0.f, __half2float(src[i * 4 + 1])));
        float b = dev_lin_to_srgb(fmaxf(0.f, __half2float(src[i * 4 + 2])));
        float a = __half2float(src[i * 4 + 3]);
        dst[i * 4 + 0] = (uint8_t)(r >= 1.f ? 255 : r * 255.0f + 0.5f);
        dst[i * 4 + 1] = (uint8_t)(gg >= 1.f ? 255 : gg * 255.0f + 0.5f);
        dst[i * 4 + 2] = (uint8_t)(b >= 1.f ? 255 : b * 255.0f + 0.5f);
        dst[i * 4 + 3] = (uint8_t)(a <= 0.f ? 0 : (a >= 1.f ? 255 : a * 255.0f + 0.5f));
    }
}

// ---- Lanczos-4 fallback (kept from the previous revision) ------------------
__device__ __forceinline__ float sinc_pi(float x) {
    if (x == 0.f) return 1.f;
    float a = (float)M_PI * x;
    return sinf(a) / a;
}
__device__ __forceinline__ float lanczos4(float x) {
    x = fabsf(x);
    if (x >= 4.f) return 0.f;
    return sinc_pi(x) * sinc_pi(x * 0.25f);
}
__global__ void k_upscale_rgba(const uint8_t *__restrict__ src,
                               uint8_t *__restrict__ dst,
                               int sw, int sh, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const float sx = (x + 0.5f) * (float)sw / (float)dw - 0.5f;
    const float sy = (y + 0.5f) * (float)sh / (float)dh - 0.5f;
    const int ix = (int)floorf(sx), iy = (int)floorf(sy);
    const float fx = sx - (float)ix, fy = sy - (float)iy;
    const int span = 9;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int oy = -span; oy <= span; ++oy) {
        float ty = (float)oy - fy;
        if (fabsf(ty) >= 4.f) continue;
        float wy = lanczos4(ty);
        if (wy == 0.f) continue;
        int cyy = iy + oy;
        if (cyy < 0) cyy = 0; else if (cyy >= sh) cyy = sh - 1;
        for (int ox = -span; ox <= span; ++ox) {
            float tx = (float)ox - fx;
            if (fabsf(tx) >= 4.f) continue;
            float w = lanczos4(tx) * wy;
            if (w == 0.f) continue;
            int cxx = ix + ox;
            if (cxx < 0) cxx = 0; else if (cxx >= sw) cxx = sw - 1;
            const uint8_t *px = src + (size_t)(cyy * sw + cxx) * 4u;
            acc[0] += w * (float)px[0]; acc[1] += w * (float)px[1];
            acc[2] += w * (float)px[2]; acc[3] += w * (float)px[3];
        }
    }
    uint8_t *o = dst + (size_t)(y * dw + x) * 4u;
    for (int c = 0; c < 4; ++c)
        o[c] = (uint8_t)(acc[c] < 0.f ? 0.f : (acc[c] > 255.f ? 255.f : acc[c]));
}

// ---- NGX helpers (no exit() calls: library context) ------------------------
static int quality_for_ratio(float r) {
    if (r < 1.2f) return QUALITY_DLAA;
    if (r < 1.6f) return QUALITY_MAXQUALITY;
    if (r < 1.85f) return QUALITY_BALANCED;
    if (r < 2.5f) return QUALITY_MAXPERF;
    return QUALITY_ULTRAPERF;
}

static const char *hint_param_for_quality(int q) {
    switch (q) {
    case QUALITY_DLAA:      return NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA;
    case QUALITY_MAXQUALITY:return NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality;
    case QUALITY_BALANCED:  return NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced;
    case QUALITY_MAXPERF:   return NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance;
    default:                return NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance;
    }
}

static bool ngx_init_once() {
    // Resolve <repo>/DLSS/lib/Linux_aarch64/rel from this .so's location so
    // installs anywhere on disk still find the DLSS plugin.
    Dl_info info;
    std::string dir = ".";
    if (dladdr((const void *)dlss_init, &info) && info.dli_fname) {
        std::string p = info.dli_fname;
        size_t s = p.find_last_of('/');
        if (s != std::string::npos) dir = p.substr(0, s);
    }
    static wchar_t wrel[4096], wdir[4096], wenv[4096];
    mbstowcs(wrel, (dir + "/../../DLSS/lib/Linux_aarch64/rel").c_str(), 4095);
    mbstowcs(wdir, dir.c_str(), 4095);
    const wchar_t *paths[4];
    int k = 0;
    paths[k++] = wrel;
    paths[k++] = wdir;
    if (const char *env = getenv("DLSS_NGX_PLUGIN_DIR")) {
        if (env[0]) {
            mbstowcs(wenv, env, 4095);
            paths[k++] = wenv;
        }
    }
    static wchar_t wdot[] = L".";
    paths[k++] = wdot;

    NVSDK_NGX_PathListInfo pli = { paths, (unsigned int)k };
    NVSDK_NGX_FeatureCommonInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.PathListInfo = pli;
    mkdir("/tmp/nvngx", 0755);
    NVSDK_NGX_Result r =
        NVSDK_NGX_CUDA_Init(0xDEADEAFULL, L"/tmp/nvngx", &fci, NVSDK_NGX_Version_API);
    if (r != NVSDK_NGX_Result_Success) {
        set_err("NVSDK_NGX_CUDA_Init failed");
        return false;
    }
    // Capability gate: SR must be present (ISR is not, on DGX Spark).
    NVSDK_NGX_Parameter *cp = nullptr;
    if (NVSDK_NGX_CUDA_GetCapabilityParameters(&cp) == NVSDK_NGX_Result_Success && cp) {
        int avail = 0;
        cp->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &avail);
        NVSDK_NGX_CUDA_DestroyParameters(cp);
        if (!avail) {
            set_err("DLSS SuperSampling not available on this GPU/driver");
            NVSDK_NGX_CUDA_Shutdown();
            return false;
        }
    }
    return true;
}

static void feat_free(NgxFeat *f) {
    if (f->feat) { NVSDK_NGX_CUDA_ReleaseFeature(f->feat); f->feat = nullptr; }
    if (f->tC) cuTexObjectDestroy(f->tC);
    if (f->tM) cuTexObjectDestroy(f->tM);
    if (f->tD) cuTexObjectDestroy(f->tD);
    if (f->sO) cuSurfObjectDestroy(f->sO);
    if (f->aC) cuArrayDestroy(f->aC);
    if (f->aO) cuArrayDestroy(f->aO);
    if (f->aM) cuArrayDestroy(f->aM);
    if (f->aD) cuArrayDestroy(f->aD);
    if (f->d_in8) cudaFree(f->d_in8);
    if (f->d_in16) cudaFree(f->d_in16);
    if (f->d_mv) cudaFree(f->d_mv);
    if (f->d_out16) cudaFree(f->d_out16);
    if (f->d_out8) cudaFree(f->d_out8);
    // NOTE: the NVSDK_NGX_Parameter map is deliberately leaked here (one tiny
    // map per dims-key): the released feature may still cache references to
    // the map it was created with, so destroying it is riskier than leaking.
    f->params = nullptr;
    *f = NgxFeat();
}

static bool feat_ensure(int sw, int sh, int dw, int dh, int quality, int preset) {
    NgxFeat *f = &g.dlss;
    if (f->active && f->in_w == sw && f->in_h == sh &&
        f->out_w == dw && f->out_h == dh &&
        f->quality == quality && f->preset == preset)
        return true;
    feat_free(f);

    if (cudaMalloc((void **)&f->d_in8, (size_t)sw * sh * 4) != cudaSuccess) return false;
    if (cudaMalloc((void **)&f->d_in16, (size_t)sw * sh * 8) != cudaSuccess) return false;
    if (cudaMalloc((void **)&f->d_mv, (size_t)sw * sh * 4) != cudaSuccess) return false;
    if (cudaMalloc((void **)&f->d_out16, (size_t)dw * dh * 8) != cudaSuccess) return false;
    if (cudaMalloc((void **)&f->d_out8, (size_t)dw * dh * 4) != cudaSuccess) return false;

    CUDA_ARRAY3D_DESCRIPTOR ad = {};
    ad.Width = sw; ad.Height = sh;
    ad.Format = CU_AD_FORMAT_HALF; ad.NumChannels = 4;
    if (cuArray3DCreate(&f->aC, &ad) != CUDA_SUCCESS) return false;
    ad.Width = dw; ad.Height = dh;
    if (cuArray3DCreate(&f->aO, &ad) != CUDA_SUCCESS) return false;
    ad.Width = sw; ad.Height = sh; ad.NumChannels = 2;
    if (cuArray3DCreate(&f->aM, &ad) != CUDA_SUCCESS) return false;
    if (cudaMemset(f->d_mv, 0, (size_t)sw * sh * 4) != cudaSuccess) return false;
    {
        CUDA_MEMCPY3D cp = {};
        cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cp.srcDevice = (CUdeviceptr)f->d_mv;
        cp.srcPitch = sw * 4; cp.srcHeight = sh;
        cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        cp.dstArray = f->aM;
        cp.WidthInBytes = sw * 4; cp.Height = sh; cp.Depth = 1;
        if (cuMemcpy3D(&cp) != CUDA_SUCCESS) return false;
    }
    ad.Format = CU_AD_FORMAT_FLOAT; ad.NumChannels = 1;
    if (cuArray3DCreate(&f->aD, &ad) != CUDA_SUCCESS) return false;
    {
        size_t n = (size_t)sw * sh;
        float *h = (float *)malloc(n * 4);
        if (!h) return false;
        for (size_t i = 0; i < n; ++i) h[i] = 1.0f;
        void *tmp = nullptr;
        bool ok = cudaMalloc(&tmp, n * 4) == cudaSuccess &&
                  cudaMemcpy(tmp, h, n * 4, cudaMemcpyHostToDevice) == cudaSuccess;
        free(h);
        if (!ok) { if (tmp) cudaFree(tmp); return false; }
        CUDA_MEMCPY3D cp = {};
        cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cp.srcDevice = (CUdeviceptr)tmp;
        cp.srcPitch = sw * 4; cp.srcHeight = sh;
        cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        cp.dstArray = f->aD;
        cp.WidthInBytes = sw * 4; cp.Height = sh; cp.Depth = 1;
        CUresult cr = cuMemcpy3D(&cp);
        cudaFree(tmp);
        if (cr != CUDA_SUCCESS) return false;
    }

    CUDA_RESOURCE_DESC rd = {};
    rd.resType = CU_RESOURCE_TYPE_ARRAY;
    CUDA_TEXTURE_DESC td = {};
    td.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    td.filterMode = CU_TR_FILTER_MODE_POINT;
    td.flags = 0;
    rd.res.array.hArray = f->aC;
    if (cuTexObjectCreate(&f->tC, &rd, &td, nullptr) != CUDA_SUCCESS) return false;
    rd.res.array.hArray = f->aM;
    if (cuTexObjectCreate(&f->tM, &rd, &td, nullptr) != CUDA_SUCCESS) return false;
    rd.res.array.hArray = f->aD;
    if (cuTexObjectCreate(&f->tD, &rd, &td, nullptr) != CUDA_SUCCESS) return false;
    CUDA_RESOURCE_DESC rds = {};
    rds.resType = CU_RESOURCE_TYPE_ARRAY;
    rds.res.array.hArray = f->aO;
    if (cuSurfObjectCreate(&f->sO, &rds) != CUDA_SUCCESS) return false;

    if (NVSDK_NGX_CUDA_AllocateParameters(&f->params) != NVSDK_NGX_Result_Success ||
        !f->params)
        return false;
    f->params->Set(NVSDK_NGX_Parameter_Width, (unsigned int)sw);
    f->params->Set(NVSDK_NGX_Parameter_Height, (unsigned int)sh);
    f->params->Set(NVSDK_NGX_Parameter_OutWidth, (unsigned int)dw);
    f->params->Set(NVSDK_NGX_Parameter_OutHeight, (unsigned int)dh);
    f->params->Set(NVSDK_NGX_Parameter_PerfQualityValue, quality);
    f->params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                   (int)NVSDK_NGX_DLSS_Feature_Flags_None);
    f->params->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
    if (preset != 0)
        f->params->Set(hint_param_for_quality(quality), preset);

    NVSDK_NGX_Result cr =
        NVSDK_NGX_CUDA_CreateFeature(NVSDK_NGX_Feature_SuperSampling,
                                     f->params, &f->feat);
    if (cr != NVSDK_NGX_Result_Success || !f->feat) return false;

    f->params->Set(NVSDK_NGX_Parameter_Color, &f->tC);
    f->params->Set(NVSDK_NGX_Parameter_Output, &f->sO);
    f->params->Set(NVSDK_NGX_Parameter_Depth, &f->tD);
    f->params->Set(NVSDK_NGX_Parameter_MotionVectors, &f->tM);
    f->params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
    f->params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
    f->params->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);
    f->params->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    f->params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    f->params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
                    (unsigned int)sw);
    f->params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
                    (unsigned int)sh);
    f->params->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
    f->params->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);

    f->in_w = sw; f->in_h = sh; f->out_w = dw; f->out_h = dh;
    f->quality = quality; f->preset = preset;
    f->active = true;
    return true;
}

// Run one DLSS-SR frame. mv_f16 may be null (zero motion). Returns true on
// neural success.
static bool dlss_eval_frame(const void *in8, const void *mv_f16_or_null,
                            int reset, void *out8,
                            int sw, int sh, int dw, int dh) {
    NgxFeat *f = &g.dlss;
    size_t in_px = (size_t)sw * sh;
    size_t out_px = (size_t)dw * dh;

    if (cudaMemcpyAsync(f->d_in8, in8, in_px * 4, cudaMemcpyHostToDevice,
                        g.stream) != cudaSuccess)
        return false;
    {
        const int blk = 256;
        int grid = (int)((in_px + blk - 1) / blk);
        if (grid > 65535) grid = 65535;
        k_rgba8_to_rgba16f<<<grid, blk, 0, g.stream>>>(
            (const uint8_t *)f->d_in8, f->d_in16, in_px);
        if (cudaGetLastError() != cudaSuccess) return false;
    }
    if (mv_f16_or_null) {
        if (cudaMemcpyAsync(f->d_mv, mv_f16_or_null, in_px * 4,
                            cudaMemcpyHostToDevice, g.stream) != cudaSuccess)
            return false;
    } else {
        if (cudaMemsetAsync(f->d_mv, 0, in_px * 4, g.stream) != cudaSuccess)
            return false;
    }
    // Device-wide sync: the cuMemcpy3D calls below are synchronous driver
    // calls ordered against the NULL stream, while our uploads/kernels run
    // on a non-blocking stream. NGX likewise runs on its own internal
    // stream (hence device- (not stream-) scoped syncs on this path).
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    {
        CUDA_MEMCPY3D cp = {};
        cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cp.srcDevice = (CUdeviceptr)f->d_in16;
        cp.srcPitch = sw * 8; cp.srcHeight = sh;
        cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        cp.dstArray = f->aC;
        cp.WidthInBytes = sw * 8; cp.Height = sh; cp.Depth = 1;
        if (cuMemcpy3D(&cp) != CUDA_SUCCESS) return false;
    }
    {
        CUDA_MEMCPY3D cp = {};
        cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cp.srcDevice = (CUdeviceptr)f->d_mv;
        cp.srcPitch = sw * 4; cp.srcHeight = sh;
        cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        cp.dstArray = f->aM;
        cp.WidthInBytes = sw * 4; cp.Height = sh; cp.Depth = 1;
        if (cuMemcpy3D(&cp) != CUDA_SUCCESS) return false;
    }

    f->params->Set(NVSDK_NGX_Parameter_Reset, reset ? 1 : 0);
    if (NVSDK_NGX_CUDA_EvaluateFeature(f->feat, f->params, NULL) !=
        NVSDK_NGX_Result_Success)
        return false;
    // Device-wide: waits for NGX's internal stream as well as ours.
    if (cudaDeviceSynchronize() != cudaSuccess) return false;

    {
        CUDA_MEMCPY3D cp = {};
        cp.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        cp.srcArray = f->aO;
        cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cp.dstDevice = (CUdeviceptr)f->d_out16;
        cp.dstPitch = dw * 8; cp.dstHeight = dh;
        cp.WidthInBytes = dw * 8; cp.Height = dh; cp.Depth = 1;
        if (cuMemcpy3D(&cp) != CUDA_SUCCESS) return false;
    }
    {
        const int blk = 256;
        int grid = (int)((out_px + blk - 1) / blk);
        if (grid > 65535) grid = 65535;
        k_rgba16f_to_rgba8<<<grid, blk, 0, g.stream>>>(
            f->d_out16, (uint8_t *)f->d_out8, out_px);
        if (cudaGetLastError() != cudaSuccess) return false;
    }
    if (cudaMemcpyAsync(out8, f->d_out8, out_px * 4, cudaMemcpyDeviceToHost,
                        g.stream) != cudaSuccess)
        return false;
    return cudaStreamSynchronize(g.stream) == cudaSuccess;
}

// Lanczos fallback (previous revision's path, kept for robustness).
static bool lanczos_upscale(const void *in, int sw, int sh,
                            void *out, int dw, int dh) {
    size_t in_bytes = (size_t)sw * sh * 4;
    size_t out_bytes = (size_t)dw * dh * 4;
    void *d_in = nullptr, *d_out = nullptr;
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess) return false;
    if (cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
        cudaFree(d_in);
        return false;
    }
    bool ok = false;
    if (cudaMemcpyAsync(d_in, in, in_bytes, cudaMemcpyHostToDevice,
                        g.stream) == cudaSuccess) {
        dim3 blk(16, 16);
        dim3 grid((dw + blk.x - 1) / blk.x, (dh + blk.y - 1) / blk.y);
        k_upscale_rgba<<<grid, blk, 0, g.stream>>>(
            (const uint8_t *)d_in, (uint8_t *)d_out, sw, sh, dw, dh);
        ok = cudaGetLastError() == cudaSuccess &&
             cudaStreamSynchronize(g.stream) == cudaSuccess &&
             cudaMemcpyAsync(out, d_out, out_bytes, cudaMemcpyDeviceToHost,
                             g.stream) == cudaSuccess &&
             cudaStreamSynchronize(g.stream) == cudaSuccess;
    }
    cudaFree(d_in);
    cudaFree(d_out);
    return ok;
}

// ---------------------------------------------------------------------------
// C ABI  (all functions are extern "C"; sizes are bytes; pixels are RGBA8)
// ---------------------------------------------------------------------------
extern "C" {

int32_t dlss_init(int32_t app_id) {
    pthread_mutex_lock(&g_mu);
    if (g.initialised) {
        pthread_mutex_unlock(&g_mu);
        return DLSS_OK;
    }
    int devCount = 0;
    cudaError_t e = cudaGetDeviceCount(&devCount);
    if (e != cudaSuccess || devCount == 0) {
        set_err("no CUDA device");
        pthread_mutex_unlock(&g_mu);
        return DLSS_ERR_NOT_INIT;
    }
    // A current context is mandatory BEFORE NGX init (else it exits).
    if (cudaSetDevice(0) != cudaSuccess ||
        cudaFree(nullptr) != cudaSuccess) {
        set_err("cannot establish CUDA context");
        pthread_mutex_unlock(&g_mu);
        return DLSS_ERR_CUDA;
    }
    if (cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking) !=
        cudaSuccess) {
        set_err("cannot create CUDA stream");
        pthread_mutex_unlock(&g_mu);
        return DLSS_ERR_CUDA;
    }
    (void)app_id;
    // NGX is best-effort: failure degrades to the Lanczos fallback.
    g.ngx_available = ngx_init_once();
    if (!g.ngx_available)
        set_err("NGX unavailable, Lanczos fallback active");
    g.initialised = true;
    g.last_path = 0;
    pthread_mutex_unlock(&g_mu);
    return DLSS_OK;
}

int32_t dlss_shutdown(void) {
    pthread_mutex_lock(&g_mu);
    if (g.initialised) {
        feat_free(&g.dlss);
        if (g.ngx_available) {
            NVSDK_NGX_CUDA_Shutdown();
            g.ngx_available = false;
        }
        if (g.stream) {
            cudaStreamDestroy(g.stream);
            g.stream = nullptr;
        }
        g.initialised = false;
    }
    pthread_mutex_unlock(&g_mu);
    return DLSS_OK;
}

int32_t dlss_is_available(void) { return g.initialised ? 1 : 0; }

int32_t dlss_last_path(void) { return g.last_path; }

int32_t dlss_get_last_error(char *buf, int32_t cap) {
    if (cap > 0) {
        size_t n = strlen(g_last_error);
        if (n >= (size_t)cap) n = (size_t)cap - 1;
        memcpy(buf, g_last_error, n);
        buf[n] = 0;
    }
    return 0;
}

int32_t dlss_supported_scale_factor(int32_t factor_x1000) {
    if (factor_x1000 == 1000 || factor_x1000 == 1500 ||
        factor_x1000 == 1724 || factor_x1000 == 2000 || factor_x1000 == 3000)
        return 1;
    return 0;
}

int32_t dlss_set_enhancement(int32_t intensity_x10,
                             int32_t local_tone_x10,
                             int32_t local_structure_x10,
                             int32_t skin_structure_x10) {
    g.enh_intensity = intensity_x10;
    g.enh_tone = local_tone_x10;
    g.enh_struct = local_structure_x10;
    (void)skin_structure_x10;
    return DLSS_OK;
}

// Full temporal entrypoint. mv_f16: host HxWx2 float16 motion (may be NULL
// for zero motion). reset: nonzero resets DLSS temporal history. quality:
// NVSDK enum (0/1/2/3/5) or -1 to derive from the ratio. preset: 0 (driver
// default) or 10/11/12/13 (J/K/L/M) applied to the active quality's hint.
int32_t dlss_upscale_rgba_ex(const void *in, int32_t in_w, int32_t in_h,
                             const void *mv_f16, int32_t reset,
                             void *out, int32_t out_w, int32_t out_h,
                             int32_t scale_x1000, int32_t quality,
                             int32_t preset) {
    if (!g.initialised) { set_err("not initialised"); return DLSS_ERR_NOT_INIT; }
    if (!in || !out || in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) {
        set_err("bad dimensions/pointer");
        return DLSS_ERR_BAD_INPUT;
    }
    if (out_w > 8192 || out_h > 8192) {
        set_err("output exceeds 8192 on an edge");
        return DLSS_ERR_BAD_INPUT;
    }
    if (quality < 0)
        quality = quality_for_ratio((float)out_w / (float)in_w);
    if (preset < 0) preset = 0;
    (void)scale_x1000; // output dims are explicit; kept for ABI compatibility

    pthread_mutex_lock(&g_mu);
    int32_t rc = DLSS_ERR_CUDA;
    bool neural = false;
    if (g.ngx_available && feat_ensure(in_w, in_h, out_w, out_h, quality, preset)) {
        if (dlss_eval_frame(in, mv_f16, reset, out, in_w, in_h, out_w, out_h)) {
            neural = true;
            rc = DLSS_OK;
        } else {
            set_err("DLSS evaluate failed, Lanczos fallback active");
        }
    } else if (g.ngx_available) {
        set_err("DLSS feature setup failed, Lanczos fallback active");
    }
    if (!neural) {
        if (lanczos_upscale(in, in_w, in_h, out, out_w, out_h))
            rc = DLSS_OK;
        else {
            set_err("Lanczos fallback failed");
            rc = DLSS_ERR_CUDA;
        }
    }
    g.last_path = neural ? 1 : 0;
    pthread_mutex_unlock(&g_mu);
    return rc;
}

// Legacy entrypoint: stills (zero motion, history reset), quality derived.
int32_t dlss_upscale_rgba(const void *in, int32_t in_w, int32_t in_h,
                          void *out, int32_t out_w, int32_t out_h,
                          int32_t scale_x1000) {
    return dlss_upscale_rgba_ex(in, in_w, in_h, nullptr, 1, out, out_w, out_h,
                                scale_x1000, -1, 0);
}

} // extern "C"
