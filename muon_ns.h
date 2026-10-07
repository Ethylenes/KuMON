#pragma once
#include <cstddef>
#include <cstdint>

// scratch space needed for [m x n] input
size_t muon_ns_workspace_bytes(int64_t m, int64_t n);

int muon_ns_forward(void* cublas_handle, void* cuda_stream, const float* x, float* out, void* ws,
                    int64_t m, int64_t n, int steps, float out_scale);
