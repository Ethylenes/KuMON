#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "muon_ns.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static cublasHandle_t get_handle() {
    static thread_local cublasHandle_t h = nullptr;
    if (h == nullptr) cublasCreate(&h);
    return h;
}

static ffi::Error MuonNsImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> x, ffi::ResultBuffer<ffi::F32> out, int64_t steps) {
    auto dims = x.dimensions();
    if(dims.size() != 2) 
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "muon_ns needs a 2D input");
    const int64_t m = dims[0], n = dims[1];
    if (m > n)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "muon_ns expects m <= n");
    
    void* ws = nullptr;
    const size_t ws_bytes = muon_ns_workspace_bytes(m, n);
    if(cudaMallocAsync(&ws, ws_bytes, stream) != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, "workspace alloc failed");
    
    int rc = muon_ns_forward((void*)get_handle(), (void*)stream, x.typed_data(), out->typed_data(), ws, m, n, (int)steps, 1.0f);
    cudaFreeAsync(ws, stream);
    if (rc != 0) return ffi::Error(ffi::ErrorCode::kInternal, "muon_ns_forward failed");
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MuonNs, MuonNsImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("steps")
);
