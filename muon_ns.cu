#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "muon_ns.h"

namespace {
constexpr float kCoefA = 3.4445f;
constexpr float kCoefB = -4.7750f;
constexpr float kCoefC = 2.0315f;
constexpr size_t kAlign = 256;

inline size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

__global__ void sumsq_(const float* __restrict__ x, int64_t n, float* __restrict__ out) {
    float s = 0.f;
    for(int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float v = x[i];
        s += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) s+= __shfl_down_sync(0xffffffffu, s, o); // max unsigned integer
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = (lane < (blockDim.x >> 5)) ? sh[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if(lane == 0) atomicAdd(out, s);
    }
}

__global__ void normalize_cast_(const float* __restrict__ x, __half* __restrict__ y, int64_t n, const float* __restrict__ sumsq) {
    const float inv = 1.0f / (sqrtf(*sumsq) + 1e-7f);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) 
        y[i] = __float2half(x[i] * inv);
}

__global__ void cast_scale_(const __half* __restrict__ x, float* __restrict__ y, int64_t n, float scale) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) y[i] = __half2float(x[i]) * scale;
}

inline int blocks_for(int64_t n, int threads) {
    int64_t b = (n + threads - 1) / threads;
    return (int)std::min<int64_t>(std::max<int64_t>(b,1), 2048);
}

// Row-major GEMM on top of column-major cuBLAS:
//  C(MxN) = alpha * op(A)(MxK) * op(B)(KxN) + beta * C
inline cublasStatus_t gemm_rm(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB, 
int M, int N, int K, float alpha, const __half* A, int lda, const __half* B, int ldb, float beta, __half* C, int ldc) {
    return cublasGemmEx(h, opB, opA, N, M, K, &alpha, B, CUDA_R_16F, ldb, A, CUDA_R_16F, lda, &beta, C, CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

} // namespace

size_t muon_ns_workspace_bytes(int64_t m, int64_t n) {
    size_t half_sz = sizeof(__half);
    return align_up(sizeof(float)) + 2 * align_up(m * n * half_sz) + 2 * align_up(m * m * half_sz);
    // sum of squares; X ping-pong; and A, B
}

int muon_ns_forward(void* cublas_handle, void* cuda_stream, const float* x, float* out, void* ws, int64_t m, int64_t n, int steps, float out_scale) {
    if (!cublas_handle || !x || !out || !ws || m <= 0 || n <= 0 || m > n || steps < 0)
        return 1;
    cublasHandle_t h = (cublasHandle_t)cublas_handle;
    cudaStream_t s = (cudaStream_t)cuda_stream;

    // Carve workspace
    char* p = (char*)ws;
    float* ss = (float*)p;
    p += align_up(sizeof(float));          __half* Xa = (__half*)p;
    p += align_up(m * n * sizeof(__half)); __half* Xb = (__half*)p;
    p += align_up(m * n * sizeof(__half)); __half* Abuf = (__half*)p;
    p += align_up(m * m * sizeof(__half)); __half* Bbuf = (__half*)p;

    const int64_t mn = m * n, mm = m * m;
    const int T = 256;

    // normalize and cast
    if (cudaMemsetAsync(ss, 0, sizeof(float), s) != cudaSuccess) return 3;
    sumsq_<<<blocks_for(mn, T), T, 0, s>>>(x, mn, ss);
    normalize_cast_<<<blocks_for(mn, T), T, 0, s>>>(x, Xa, mn, ss);

    // iterations
    if (cublasSetStream(h, s) != CUBLAS_STATUS_SUCCESS) return 2;
    __half* cur = Xa;
    __half* nxt = Xb;
    for (int it = 0; it < steps; ++it) {
        // A = X X^T (m x m), K = n
        if (gemm_rm(h, CUBLAS_OP_N, CUBLAS_OP_T, (int)m, (int)m, (int)n, 1.f, cur, (int)n, cur, (int)n, 0.f, Abuf, (int)m) != CUBLAS_STATUS_SUCCESS)
            return 2;
        // B = A ; B = c*A@A + b*B
        if (cudaMemcpyAsync(Bbuf, Abuf, mm * sizeof(__half), cudaMemcpyDeviceToDevice, s) != cudaSuccess)
            return 3;
        if (gemm_rm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)m, (int)m, (int)m, kCoefC, Abuf, (int)m, Abuf, (int)m, kCoefB, Bbuf, (int)m) != CUBLAS_STATUS_SUCCESS)
            return 2;
        // Xn = X ; Xn = B@X + a*Xn
        if (cudaMemcpyAsync(nxt, cur, mn * sizeof(__half), cudaMemcpyDeviceToDevice, s) != cudaSuccess)
            return 3;
        if (gemm_rm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)m, (int)n, (int)m, 1.f, Bbuf, (int)m, cur, (int)n, kCoefA, nxt, (int)n) != CUBLAS_STATUS_SUCCESS)
            return 2;
        std::swap(cur, nxt);
    }

    // cast back
    cast_scale_<<<blocks_for(mn, T), T, 0, s>>>(cur, out, mn, out_scale);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
