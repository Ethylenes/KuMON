#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "muon_ns.h"


int64_t workspace_bytes(int64_t m, int64_t n) {
  return (int64_t)muon_ns_workspace_bytes(m, n);
}

void forward(torch::Tensor x, torch::Tensor out, torch::Tensor ws, int64_t steps) {
    TORCH_CHECK(x.is_cuda() && out.is_cuda() && ws.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && out.scalar_type() == torch::kFloat32, "x and out must be float32");
    TORCH_CHECK(ws.scalar_type() == torch::kUInt8, "ws must be uint8");
    TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && out.is_contiguous(), "x must be a contiguous 2D tensor");
    const int64_t m = x.size(0), n = x.size(1);
    TORCH_CHECK(m <= n, "expected m <= n (transpose tall matrices first)");
    TORCH_CHECK(ws.numel() >= (int64_t)muon_ns_workspace_bytes(m, n), "workspace too small");

    c10::cuda::CUDAGuard guard(x.device());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int rc = muon_ns_forward((void*)handle, (void*)stream, x.data_ptr<float>(), out.data_ptr<float>(), ws.data_ptr(), m, n, (int)steps, 1.0f);
    TORCH_CHECK(rc == 0, "muon_ns_forward failed with code ", rc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("workspace_bytes", &workspace_bytes, "scratch bytes for (m, n)");
    m.def("forward", &forward, "fused Newton-Schulz (CUDA)");
}