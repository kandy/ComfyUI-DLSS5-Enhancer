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
#include <vector>
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_defs.h"

// NVIDIA Optical Flow Accelerator (NVOFA) -- MIT-licensed SDK headers, vendored
// under include/nvofapi. Used for flow-guided frame interpolation.
#include "nvOpticalFlowCuda.h"
#include "nvOpticalFlowCommon.h"

// Forward declarations (dladdr uses dlss_init; interpolation refines with SR).
extern "C" int32_t dlss_init(int32_t app_id);
extern "C" int32_t dlss_upscale_rgba(const void *in, int32_t in_w, int32_t in_h,
                                     void *out, int32_t out_w, int32_t out_h,
                                     int32_t scale_x1000);

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

// ---- Optical-flow interpolation state --------------------------------------
// NVOFA is loaded at runtime from the installed driver; the SDK headers only
// define the interface. Everything degrades to a cross-fade when it is missing.
struct OfSession {
    bool ready = false;
    int w = 0, h = 0;
    NvOFHandle of = nullptr;
    NV_OF_CUDA_API_FUNCTION_LIST fn = {};
    NvOFGPUBufferHandle bufA = nullptr, bufB = nullptr;
    NvOFGPUBufferHandle bufF = nullptr, bufW = nullptr;
    NvOFGPUBufferHandle bufFPrev = nullptr, bufWPrev = nullptr;
    CUdeviceptr pA = 0, pB = 0, pF = 0, pW = 0;
    CUdeviceptr pFPrev = 0, pWPrev = 0;
    uint32_t pitch_in = 0, pitch_out = 0;
    uint8_t *d_out = nullptr; // w*h*4 warp target
};

struct InterpState {
    int available = -1; // -1 = not probed, 1 = NVOFA usable, 0 = fallback
    int last_path = 0;  // 1 = optical flow, 0 = linear cross-fade
    int last_order = 0; // 2 = N-2/N-1/N acceleration-aware, 1 = pairwise, 0 = blend
    void *lib = nullptr;
    NV_OF_STATUS (*create)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST *) = nullptr;
};

static InterpState g_interp;
static OfSession g_of;
static pthread_mutex_t g_of_mu = PTHREAD_MUTEX_INITIALIZER;

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

// ---- Flow-guided interpolation kernels -------------------------------------
// NVOFA returns S10.5 fixed-point vectors: 1/32 pixel per unit.
__device__ __forceinline__ float2 dev_flow_at(const int16_t *f, int fw, int fh,
                                              int pitch_px, float x, float y) {
    x = fminf(fmaxf(x, 0.f), (float)(fw - 1));
    y = fminf(fmaxf(y, 0.f), (float)(fh - 1));
    const int x0 = (int)floorf(x), y0 = (int)floorf(y);
    const int x1 = min(x0 + 1, fw - 1), y1 = min(y0 + 1, fh - 1);
    const float fx = x - (float)x0, fy = y - (float)y0;
    const int16_t *r0 = f + (size_t)y0 * pitch_px;
    const int16_t *r1 = f + (size_t)y1 * pitch_px;
    const float a0x = (float)r0[2 * x0], a0y = (float)r0[2 * x0 + 1];
    const float a1x = (float)r0[2 * x1], a1y = (float)r0[2 * x1 + 1];
    const float b0x = (float)r1[2 * x0], b0y = (float)r1[2 * x0 + 1];
    const float b1x = (float)r1[2 * x1], b1y = (float)r1[2 * x1 + 1];
    const float ax = (1.f - fx) * a0x + fx * a1x;
    const float ay = (1.f - fx) * a0y + fx * a1y;
    const float bx = (1.f - fx) * b0x + fx * b1x;
    const float by = (1.f - fx) * b0y + fx * b1y;
    return make_float2(((1.f - fy) * ax + fy * bx) * (1.f / 32.f),
                       ((1.f - fy) * ay + fy * by) * (1.f / 32.f));
}

__device__ __forceinline__ bool dev_bilinear_rgba(const uint8_t *img, int w, int h,
                                                  int pitch, float x, float y,
                                                  float out[4]) {
    if (x < 0.f || y < 0.f || x > (float)(w - 1) || y > (float)(h - 1)) return false;
    const int x0 = (int)x, y0 = (int)y;
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = x - (float)x0, fy = y - (float)y0;
    const uint8_t *r0 = img + (size_t)y0 * pitch;
    const uint8_t *r1 = img + (size_t)y1 * pitch;
    for (int c = 0; c < 4; ++c) {
        const float a = (1.f - fx) * (float)r0[x0 * 4 + c] + fx * (float)r0[x1 * 4 + c];
        const float b = (1.f - fx) * (float)r1[x0 * 4 + c] + fx * (float)r1[x1 * 4 + c];
        out[c] = (1.f - fy) * a + fy * b;
    }
    return true;
}

// Bidirectional warp of A (t=0) and B (t=1) to time t, with a forward/
// backward flow-consistency weight so disoccluded pixels take the frame that
// actually sees them.
__global__ void k_interp_warp(const uint8_t *__restrict__ A,
                              const uint8_t *__restrict__ B,
                              const int16_t *__restrict__ F,
                              const int16_t *__restrict__ Bk,
                              int w, int h, int pitch_a, int pitch_b,
                              int fpitch_px, float t,
                              uint8_t *__restrict__ out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const float2 f = dev_flow_at(F, w, h, fpitch_px, (float)x, (float)y);
    const float2 b = dev_flow_at(Bk, w, h, fpitch_px, (float)x, (float)y);

    float ca[4], cb[4];
    const bool va = dev_bilinear_rgba(A, w, h, pitch_a, (float)x - t * f.x,
                                      (float)y - t * f.y, ca);
    const bool vb = dev_bilinear_rgba(B, w, h, pitch_b, (float)x - (1.f - t) * b.x,
                                      (float)y - (1.f - t) * b.y, cb);

    const float2 b_at_a = dev_flow_at(Bk, w, h, fpitch_px,
                                      (float)x + b.x, (float)y + b.y);
    const float2 f_at_b = dev_flow_at(F, w, h, fpitch_px,
                                      (float)x + f.x, (float)y + f.y);
    const float e_a = hypotf(f.x + b_at_a.x, f.y + b_at_a.y);
    const float e_b = hypotf(b.x + f_at_b.x, b.y + f_at_b.y);
    const float sigma = 1.5f;
    const float conf_a = expf(-(e_a * e_a) / (2.f * sigma * sigma));
    const float conf_b = expf(-(e_b * e_b) / (2.f * sigma * sigma));

    float w_a = va ? (1.f - t) * (0.15f + 0.85f * conf_a) : 0.f;
    float w_b = vb ? t * (0.15f + 0.85f * conf_b) : 0.f;
    float sum = w_a + w_b;
    if (sum < 1e-4f) {
        if (va && !vb) { w_a = 1.f; w_b = 0.f; sum = 1.f; }
        else if (vb && !va) { w_a = 0.f; w_b = 1.f; sum = 1.f; }
        else { w_a = 1.f - t; w_b = t; sum = 1.f; }
    }

    uint8_t *o = out + ((size_t)y * w + x) * 4;
    for (int c = 0; c < 4; ++c) {
        float v;
        if (va && vb) v = (ca[c] * w_a + cb[c] * w_b) / sum;
        else if (va) v = ca[c];
        else if (vb) v = cb[c];
        else v = (1.f - t) * (float)A[(size_t)y * pitch_a + x * 4 + c]
               + t * (float)B[(size_t)y * pitch_b + x * 4 + c];
        o[c] = (uint8_t)fminf(fmaxf(v + 0.5f, 0.f), 255.f);
    }
}

// Acceleration-aware warp using three frames.  F is prev->next, Bk is
// next->prev and BkPrev is prev->prev2, all sampled at the output pixel in the
// prev frame.  With v0=(-BkPrev + F)/2 and acc=F+BkPrev, the trajectory
// d(t)=v0*t+0.5*acc*t^2 passes through prev at t=0 and next at t=1 while
// following the velocity and acceleration measured across the three frames.
__global__ void k_interp_warp_3f(const uint8_t *__restrict__ A,
                                 const uint8_t *__restrict__ B,
                                 const int16_t *__restrict__ F,
                                 const int16_t *__restrict__ Bk,
                                 const int16_t *__restrict__ BkPrev,
                                 int w, int h, int pitch_a, int pitch_b,
                                 int fpitch_px, float t,
                                 uint8_t *__restrict__ out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const float2 f = dev_flow_at(F, w, h, fpitch_px, (float)x, (float)y);
    const float2 b = dev_flow_at(Bk, w, h, fpitch_px, (float)x, (float)y);
    const float2 bp = dev_flow_at(BkPrev, w, h, fpitch_px, (float)x, (float)y);

    // v0 is the central-difference velocity at prev; acc is the second
    // difference.  d(0)=0 and d(1)=f, so the trajectory still lands on next.
    const float2 v0 = make_float2(0.5f * (f.x - bp.x), 0.5f * (f.y - bp.y));
    const float2 acc = make_float2(f.x + bp.x, f.y + bp.y);
    const float2 d = make_float2(v0.x * t + 0.5f * acc.x * t * t,
                                 v0.y * t + 0.5f * acc.y * t * t);

    // Displacement from next backward to time t.
    const float2 back = make_float2(f.x - d.x, f.y - d.y);

    float ca[4], cb[4];
    const bool va = dev_bilinear_rgba(A, w, h, pitch_a, (float)x - d.x,
                                      (float)y - d.y, ca);
    const bool vb = dev_bilinear_rgba(B, w, h, pitch_b, (float)x + back.x,
                                      (float)y + back.y, cb);

    const float2 b_at_a = dev_flow_at(Bk, w, h, fpitch_px,
                                      (float)x + b.x, (float)y + b.y);
    const float2 f_at_b = dev_flow_at(F, w, h, fpitch_px,
                                      (float)x + f.x, (float)y + f.y);
    const float e_a = hypotf(f.x + b_at_a.x, f.y + b_at_a.y);
    const float e_b = hypotf(b.x + f_at_b.x, b.y + f_at_b.y);
    const float sigma = 1.5f;
    const float conf_a = expf(-(e_a * e_a) / (2.f * sigma * sigma));
    const float conf_b = expf(-(e_b * e_b) / (2.f * sigma * sigma));

    float w_a = va ? (1.f - t) * (0.15f + 0.85f * conf_a) : 0.f;
    float w_b = vb ? t * (0.15f + 0.85f * conf_b) : 0.f;
    float sum = w_a + w_b;
    if (sum < 1e-4f) {
        if (va && !vb) { w_a = 1.f; w_b = 0.f; sum = 1.f; }
        else if (vb && !va) { w_a = 0.f; w_b = 1.f; sum = 1.f; }
        else { w_a = 1.f - t; w_b = t; sum = 1.f; }
    }

    uint8_t *o = out + ((size_t)y * w + x) * 4;
    for (int c = 0; c < 4; ++c) {
        float v;
        if (va && vb) v = (ca[c] * w_a + cb[c] * w_b) / sum;
        else if (va) v = ca[c];
        else if (vb) v = cb[c];
        else v = (1.f - t) * (float)A[(size_t)y * pitch_a + x * 4 + c]
               + t * (float)B[(size_t)y * pitch_b + x * 4 + c];
        o[c] = (uint8_t)fminf(fmaxf(v + 0.5f, 0.f), 255.f);
    }
}

// Linear cross-fade used when NVOFA is not available.
__global__ void k_interp_blend(const uint8_t *__restrict__ A,
                               const uint8_t *__restrict__ B,
                               int w, int h, float t,
                               uint8_t *__restrict__ out) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= (size_t)w * h * 4) return;
    for (int c = 0; c < 4; ++c) {
        const float v = (1.f - t) * (float)A[i + c] + t * (float)B[i + c];
        out[i + c] = (uint8_t)fminf(fmaxf(v + 0.5f, 0.f), 255.f);
    }
}

// ---- NVOFA session management ----------------------------------------------
static void of_release() {
    if (g_of.fn.nvOFDestroyGPUBufferCuda) {
        if (g_of.bufA) g_of.fn.nvOFDestroyGPUBufferCuda(g_of.bufA);
        if (g_of.bufB) g_of.fn.nvOFDestroyGPUBufferCuda(g_of.bufB);
        if (g_of.bufF) g_of.fn.nvOFDestroyGPUBufferCuda(g_of.bufF);
        if (g_of.bufW) g_of.fn.nvOFDestroyGPUBufferCuda(g_of.bufW);
        if (g_of.bufFPrev) g_of.fn.nvOFDestroyGPUBufferCuda(g_of.bufFPrev);
        if (g_of.bufWPrev) g_of.fn.nvOFDestroyGPUBufferCuda(g_of.bufWPrev);
    }
    if (g_of.fn.nvOFDestroy && g_of.of) g_of.fn.nvOFDestroy(g_of.of);
    if (g_of.d_out) cudaFree(g_of.d_out);
    g_of = OfSession();
}

static bool of_load_library() {
    if (g_interp.available >= 0) return g_interp.available > 0;
    g_interp.available = 0;
    static const char *names[] = {"libnvidia-opticalflow.so.1",
                                  "libnvidia-opticalflow.so", nullptr};
    for (int i = 0; names[i]; ++i) {
        g_interp.lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (g_interp.lib) break;
    }
    if (!g_interp.lib) return false;
    auto create = (NV_OF_STATUS (*)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST *))
        dlsym(g_interp.lib, "NvOFAPICreateInstanceCuda");
    if (!create) return false;
    auto maxv = (NV_OF_STATUS (*)(uint32_t *))
        dlsym(g_interp.lib, "NvOFGetMaxSupportedApiVersion");
    if (maxv) {
        uint32_t supported = 0;
        if (maxv(&supported) == NV_OF_SUCCESS &&
            (supported >> 4) < (NV_OF_API_VERSION >> 4))
            return false;
    }
    g_interp.create = create;
    g_interp.available = 1;
    return true;
}

static bool of_execute_flow_into(const uint8_t *prev, const uint8_t *next,
                                 int w, int h, uint32_t stride_in,
                                 uint32_t stride_out,
                                 NvOFGPUBufferHandle dstF,
                                 NvOFGPUBufferHandle dstW,
                                 CUdeviceptr pF, CUdeviceptr pW) {
    if (cudaMemcpy2D((void *)g_of.pA, stride_in, prev, (size_t)w * 4,
                     (size_t)w * 4, h, cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    if (cudaMemcpy2D((void *)g_of.pB, stride_in, next, (size_t)w * 4,
                     (size_t)w * 4, h, cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    if (cudaMemset2D((void *)pF, stride_out, 0, (size_t)w * 4, h) != cudaSuccess)
        return false;
    if (cudaMemset2D((void *)pW, stride_out, 0, (size_t)w * 4, h) != cudaSuccess)
        return false;

    NV_OF_EXECUTE_INPUT_PARAMS in;
    memset(&in, 0, sizeof(in));
    in.inputFrame = g_of.bufA;
    in.referenceFrame = g_of.bufB;
    in.disableTemporalHints = NV_OF_TRUE; // frame pairs, not a live stream
    NV_OF_EXECUTE_OUTPUT_PARAMS out;
    memset(&out, 0, sizeof(out));
    out.outputBuffer = dstF;
    out.bwdOutputBuffer = dstW;

    if (g_of.fn.nvOFExecute(g_of.of, &in, &out) != NV_OF_SUCCESS) return false;
    return cudaDeviceSynchronize() == cudaSuccess;
}

static bool of_ensure(int w, int h) {
    if (g_of.ready && g_of.w == w && g_of.h == h) return true;
    of_release();
    if (!of_load_library()) return false;

    if (g_of.fn.nvCreateOpticalFlowCuda == nullptr) {
        NV_OF_CUDA_API_FUNCTION_LIST list;
        memset(&list, 0, sizeof(list));
        if (g_interp.create(NV_OF_API_VERSION, &list) != NV_OF_SUCCESS) return false;
        g_of.fn = list;
    }
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) return false;
    if (g_of.fn.nvCreateOpticalFlowCuda(ctx, &g_of.of) != NV_OF_SUCCESS) {
        g_of.of = nullptr;
        return false;
    }

    NV_OF_INIT_PARAMS init;
    memset(&init, 0, sizeof(init));
    init.width = (uint32_t)w;
    init.height = (uint32_t)h;
    init.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_1;
    init.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_1;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = NV_OF_PERF_LEVEL_MEDIUM;
    init.enableExternalHints = NV_OF_FALSE;
    init.enableOutputCost = NV_OF_FALSE;
    init.hPrivData = nullptr;
    init.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    init.enableRoi = NV_OF_FALSE;
    init.predDirection = NV_OF_PRED_DIRECTION_BOTH;
    init.enableGlobalFlow = NV_OF_FALSE;
    init.inputBufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;
    if (g_of.fn.nvOFInit(g_of.of, &init) != NV_OF_SUCCESS) return false;

    NV_OF_BUFFER_DESCRIPTOR bd;
    memset(&bd, 0, sizeof(bd));
    bd.width = (uint32_t)w;
    bd.height = (uint32_t)h;
    bd.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;
    bd.bufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;
    if (g_of.fn.nvOFCreateGPUBufferCuda(g_of.of, &bd, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                                        &g_of.bufA) != NV_OF_SUCCESS) return false;
    if (g_of.fn.nvOFCreateGPUBufferCuda(g_of.of, &bd, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                                        &g_of.bufB) != NV_OF_SUCCESS) return false;
    bd.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
    bd.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;
    if (g_of.fn.nvOFCreateGPUBufferCuda(g_of.of, &bd, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                                        &g_of.bufF) != NV_OF_SUCCESS) return false;
    if (g_of.fn.nvOFCreateGPUBufferCuda(g_of.of, &bd, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                                        &g_of.bufW) != NV_OF_SUCCESS) return false;
    if (g_of.fn.nvOFCreateGPUBufferCuda(g_of.of, &bd, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                                        &g_of.bufFPrev) != NV_OF_SUCCESS) return false;
    if (g_of.fn.nvOFCreateGPUBufferCuda(g_of.of, &bd, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                                        &g_of.bufWPrev) != NV_OF_SUCCESS) return false;

    g_of.pA = g_of.fn.nvOFGPUBufferGetCUdeviceptr(g_of.bufA);
    g_of.pB = g_of.fn.nvOFGPUBufferGetCUdeviceptr(g_of.bufB);
    g_of.pF = g_of.fn.nvOFGPUBufferGetCUdeviceptr(g_of.bufF);
    g_of.pW = g_of.fn.nvOFGPUBufferGetCUdeviceptr(g_of.bufW);
    g_of.pFPrev = g_of.fn.nvOFGPUBufferGetCUdeviceptr(g_of.bufFPrev);
    g_of.pWPrev = g_of.fn.nvOFGPUBufferGetCUdeviceptr(g_of.bufWPrev);

    NV_OF_CUDA_BUFFER_STRIDE_INFO si;
    memset(&si, 0, sizeof(si));
    if (g_of.fn.nvOFGPUBufferGetStrideInfo(g_of.bufA, &si) != NV_OF_SUCCESS) return false;
    g_of.pitch_in = si.strideInfo[0].strideXInBytes;
    memset(&si, 0, sizeof(si));
    if (g_of.fn.nvOFGPUBufferGetStrideInfo(g_of.bufF, &si) != NV_OF_SUCCESS) return false;
    g_of.pitch_out = si.strideInfo[0].strideXInBytes;

    if (cudaMalloc((void **)&g_of.d_out, (size_t)w * h * 4) != cudaSuccess) {
        g_of.d_out = nullptr;
        return false;
    }
    g_of.fn.nvOFSetIOCudaStreams(g_of.of, (CUstream)g.stream, (CUstream)g.stream);
    g_of.w = w;
    g_of.h = h;
    g_of.ready = true;
    return true;
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
    // Try the bundled pack layout first, then the dev workspace and a copy
    // dropped directly next to the .so. Only existing directories are listed.
    static wchar_t wrel[5][4096], wdir[4096], wenv[4096];
    const std::string candidates[4] = {
        dir + "/../../DLSS/lib/Linux_aarch64/rel",                          // pack
        dir + "/../DLSS/lib/Linux_aarch64/rel",                             // sibling
        dir + "/../../ComfyUI-DLSS5-Enhancer/DLSS/lib/Linux_aarch64/rel",   // dev tree
        dir + "/DLSS/lib/Linux_aarch64/rel",                                // local copy
    };
    const wchar_t *paths[7];
    int k = 0;
    for (int i = 0; i < 4; ++i) {
        struct stat st;
        if (stat(candidates[i].c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            mbstowcs(wrel[k], candidates[i].c_str(), 4095);
            paths[k] = wrel[k];
            ++k;
        }
    }
    mbstowcs(wdir, dir.c_str(), 4095);
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
        of_release();
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

// ---------------------------------------------------------------------------
// Frame interpolation (NVIDIA Optical Flow Accelerator + DLSS refinement)
// ---------------------------------------------------------------------------
int32_t dlss_interpolate_available(void) {
    pthread_mutex_lock(&g_of_mu);
    const bool ok = of_load_library();
    pthread_mutex_unlock(&g_of_mu);
    return ok ? 1 : 0;
}

int32_t dlss_interp_last_path(void) { return g_interp.last_path; }

int32_t dlss_interp_last_order(void) { return g_interp.last_order; }

static void interp_refine_frames(void *out, int w, int h, int count) {
    // DLSS-SR at 1x (DLAA) on each synthesised frame. This takes g_mu, so it
    // must run outside g_of_mu. Failure is non-fatal: the warped frame is
    // already valid.
    const size_t frame_bytes = (size_t)w * h * 4;
    std::vector<uint8_t> tmp(frame_bytes);
    for (int k = 0; k < count; ++k) {
        uint8_t *frame = (uint8_t *)out + (size_t)k * frame_bytes;
        if (dlss_upscale_rgba(frame, w, h, tmp.data(), w, h, 1000) == DLSS_OK)
            memcpy(frame, tmp.data(), frame_bytes);
    }
}

// Generate `count` frames strictly between `prev` (t=0) and `next` (t=1).
// out receives count * w * h * 4 RGBA8 bytes, in temporal order.
// When prev2 is non-NULL the trajectory is estimated across
// prev2 -> prev -> next: velocity is the central difference at prev and
// acceleration is the second difference, so inserted frames follow curved /
// accelerating motion instead of assuming a constant pairwise velocity.
// refine != 0 runs each generated frame through the DLSS neural engine
// (DLAA, 1x) to clean warp artefacts. Falls back to a linear cross-fade when
// the optical-flow engine is unavailable; dlss_interp_last_path() reports it
// and dlss_interp_last_order() reports 2/1/0.
int32_t dlss_interpolate_rgba_3f(const void *prev2, const void *prev,
                                 const void *next, int32_t w, int32_t h,
                                 int32_t count, int32_t refine, void *out) {
    if (!g.initialised) { set_err("not initialised"); return DLSS_ERR_NOT_INIT; }
    if (!prev || !next || !out || w <= 0 || h <= 0 || count < 1 || count > 3) {
        set_err("bad interpolation arguments");
        return DLSS_ERR_BAD_INPUT;
    }
    if (w > 8192 || h > 8192) {
        set_err("interpolation frame exceeds 8192 on an edge");
        return DLSS_ERR_BAD_INPUT;
    }
    if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
        set_err("cannot establish CUDA context");
        return DLSS_ERR_CUDA;
    }

    const size_t frame_bytes = (size_t)w * h * 4;
    uint8_t *d_a = nullptr, *d_b = nullptr, *d_res = nullptr;
    int pitch_a = w * 4, pitch_b = w * 4;
    bool used_flow = false;
    bool used_second_order = false;
    bool ok = false;

    pthread_mutex_lock(&g_of_mu);
    if (of_ensure(w, h)) {
        bool previous_ok = false;
        if (prev2 != nullptr) {
            previous_ok = of_execute_flow_into(
                (const uint8_t *)prev2, (const uint8_t *)prev, w, h,
                g_of.pitch_in, g_of.pitch_out,
                g_of.bufFPrev, g_of.bufWPrev, g_of.pFPrev, g_of.pWPrev);
        }

        // Run the current pair last: g_of.pA/pB always hold prev and next for
        // the warp kernels, while the previous pair stays in its own buffers.
        bool current_ok = of_execute_flow_into(
            (const uint8_t *)prev, (const uint8_t *)next, w, h,
            g_of.pitch_in, g_of.pitch_out,
            g_of.bufF, g_of.bufW, g_of.pF, g_of.pW);

        if (current_ok) {
            d_a = (uint8_t *)g_of.pA;
            d_b = (uint8_t *)g_of.pB;
            d_res = g_of.d_out;
            pitch_a = pitch_b = (int)g_of.pitch_in;
            used_flow = true;
            used_second_order = previous_ok;
        }
    }

    if (!used_flow) {
        // No optical flow: upload both frames and cross-fade on the GPU.
        if (cudaMalloc((void **)&d_a, frame_bytes) == cudaSuccess &&
            cudaMalloc((void **)&d_b, frame_bytes) == cudaSuccess &&
            cudaMalloc((void **)&d_res, frame_bytes) == cudaSuccess) {
            if (cudaMemcpyAsync(d_a, prev, frame_bytes, cudaMemcpyHostToDevice,
                                g.stream) != cudaSuccess ||
                cudaMemcpyAsync(d_b, next, frame_bytes, cudaMemcpyHostToDevice,
                                g.stream) != cudaSuccess ||
                cudaStreamSynchronize(g.stream) != cudaSuccess) {
                cudaFree(d_a); cudaFree(d_b); cudaFree(d_res);
                d_a = d_b = d_res = nullptr;
            }
        } else {
            if (d_a) cudaFree(d_a);
            if (d_b) cudaFree(d_b);
            if (d_res) cudaFree(d_res);
            d_a = d_b = d_res = nullptr;
        }
    }

    if (d_a && d_b && d_res) {
        dim3 blk(16, 16);
        dim3 grid((w + blk.x - 1) / blk.x, (h + blk.y - 1) / blk.y);
        ok = true;
        for (int k = 1; k <= count && ok; ++k) {
            const float t = (float)k / (float)(count + 1);
            if (used_second_order) {
                k_interp_warp_3f<<<grid, blk, 0, g.stream>>>(
                    d_a, d_b, (const int16_t *)g_of.pF,
                    (const int16_t *)g_of.pW, (const int16_t *)g_of.pWPrev,
                    w, h, pitch_a, pitch_b, (int)(g_of.pitch_out / 4), t, d_res);
            } else if (used_flow) {
                k_interp_warp<<<grid, blk, 0, g.stream>>>(
                    d_a, d_b, (const int16_t *)g_of.pF,
                    (const int16_t *)g_of.pW,
                    w, h, pitch_a, pitch_b, (int)(g_of.pitch_out / 4), t, d_res);
            } else {
                const int total = w * h;
                k_interp_blend<<<((total + 255) / 256), 256, 0, g.stream>>>(
                    d_a, d_b, w, h, t, d_res);
            }
            if (cudaGetLastError() != cudaSuccess ||
                cudaStreamSynchronize(g.stream) != cudaSuccess ||
                cudaMemcpyAsync((uint8_t *)out + (size_t)(k - 1) * frame_bytes,
                                d_res, frame_bytes, cudaMemcpyDeviceToHost,
                                g.stream) != cudaSuccess ||
                cudaStreamSynchronize(g.stream) != cudaSuccess)
                ok = false;
        }
        if (!used_flow) {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_res);
        }
    }
    pthread_mutex_unlock(&g_of_mu);

    if (!ok) {
        set_err("frame interpolation failed");
        return DLSS_ERR_CUDA;
    }
    g_interp.last_path = used_flow ? 1 : 0;
    g_interp.last_order = used_second_order ? 2 : (used_flow ? 1 : 0);

    if (refine)
        interp_refine_frames(out, w, h, count);
    return DLSS_OK;
}

int32_t dlss_interpolate_rgba(const void *prev, const void *next,
                              int32_t w, int32_t h, int32_t count,
                              int32_t refine, void *out) {
    return dlss_interpolate_rgba_3f(nullptr, prev, next, w, h, count, refine, out);
}

// Legacy entrypoint: stills (zero motion, history reset), quality derived.
int32_t dlss_upscale_rgba(const void *in, int32_t in_w, int32_t in_h,
                          void *out, int32_t out_w, int32_t out_h,
                          int32_t scale_x1000) {
    return dlss_upscale_rgba_ex(in, in_w, in_h, nullptr, 1, out, out_w, out_h,
                                scale_x1000, -1, 0);
}

} // extern "C"
