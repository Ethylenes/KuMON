# KuMON (CUDA Momentum Orthogonalized by Newton-schulz)

## Structure

| File | Purpose |
|---|---|
| `muon_ns.h / muon_ns.cu` | Core CUDA code, fused normalized, cast, and scale; 5-step quintic NS with cuBLAS tensor-core GEMMs |
| `muon_torch.cpp / muon_torch.py` | PyTorch extension, reference NS, `Muon` optimizer |
| `muon_jax.cu / muon_jax.py` | XLA FFI handler, JAX reference NS, Optax integration |
| `test_correctness.py` | Fused vs reference, singular-value check, torch-vs-JAX |
| `bench.py` | Speedup (Slowdown 💀) benchmarking |
| `train_demo.py` | Demo training pipeline; Same MLP + data + init in both frameworks |

The pipeline for `muon_ns.cu` is:
1. ||x|_F^2 reduction
2. X0 = fp16(X / (||X||_F + eps))
3. Repeat steps times:

    A = X X^T (tensor-core GEMM, fp16 in / fp32 acc)

    B = A (D2D copy)

    B = c*A@A + b*B (GEMM, scale-add fusion via beta)

    Xn = X (D2D copy) 

    Xn = 1*B@X + a*Xn (GEMM, scale-add fusion via beta)
    
4. out = float(X) * out_scale (fused cast + scale kernel)

## Set Up
> Experiments were run on Google CoLab's T4 GPU

### Torch Requirements:
``` bash
pip install -q ninja
```
### JAX Requirements:
``` bash
pip uninstall -y jax-cuda13-plugin jax-cuda13-pjrt
pip install -q -U "jax[cuda12]" optax
```

## Tests and Benchmarks
```bash
cd KuMON
python test_correctness.py  # --skip-jax
python bench.py             # --skip-jax
python train_demo.py --framework <torch/jax> --impl <fused/reference>
```


## Output

`test_correctness.py`:
```
[PASS] torch  (256, 256)   rel_err=0.0030  sv in [0.158, 1.202]  max|sv-sv_ref|=0.0008
[PASS] torch  (512, 1024)  rel_err=0.0034  sv in [0.682, 1.135]  max|sv-sv_ref|=0.0008
[PASS] torch  (1024, 512)  rel_err=0.0034  sv in [0.682, 1.135]  max|sv-sv_ref|=0.0007
[PASS] torch  (300, 700)   rel_err=0.0035  sv in [0.682, 1.135]  max|sv-sv_ref|=0.0012
[PASS] torch  (768, 3072)  rel_err=0.0031  sv in [0.682, 1.134]  max|sv-sv_ref|=0.0005

[PASS] jax    (256, 256)   rel_err=0.0051  sv in [0.075, 1.202]  max|sv-sv_ref|=0.0071
[PASS] jax    (512, 1024)  rel_err=0.0056  sv in [0.682, 1.135]  max|sv-sv_ref|=0.0067
[PASS] jax    (1024, 512)  rel_err=0.0057  sv in [0.682, 1.135]  max|sv-sv_ref|=0.0071
[PASS] jax    (300, 700)   rel_err=0.0056  sv in [0.682, 1.135]  max|sv-sv_ref|=0.0071
[PASS] jax    (768, 3072)  rel_err=0.0054  sv in [0.682, 1.135]  max|sv-sv_ref|=0.0065
[PASS] torch-vs-jax fused rel diff = 0.00e+00
All Tests Passed
```

`bench.py`:
```
         shape |  torch ref  torch fused  speedup  |   jax ref  jax fused  speedup
    (512, 512) |     0.749ms      0.501ms    1.50x |    0.476ms     0.514ms    0.93x
  (1024, 1024) |     1.933ms      1.464ms    1.32x |    1.291ms     1.388ms    0.93x
   (768, 3072) |     2.320ms      1.963ms    1.18x |    1.893ms     1.891ms    1.00x
  (2048, 2048) |    12.509ms     11.808ms    1.06x |   11.506ms    11.731ms    0.98x
  (2048, 8192) |    34.195ms     33.223ms    1.03x |   32.837ms    32.950ms    1.00x
```

## Todo
- copy-free epilogue using CUTLASS
- symmetric matrix rank-k update for `X X^T` and `A@A`
- batched/grouped GEMM across layers
