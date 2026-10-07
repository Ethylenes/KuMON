import argparse

import numpy as np
import torch

SHAPES = [(256, 256), (512, 1024), (1024, 512), (300, 700), (768, 3072)]
REL_TOL = 0.08
SV_MAX = 1.5

def report(name, shape, fused, ref):
    fused = np.asarray(fused, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    rel = np.linalg.norm(fused - ref) / np.linalg.norm(ref)
    sv = np.linalg.svd(fused, compute_uv=False)
    sv_ref = np.linalg.svd(ref, compute_uv=False)
    # theoretically, singular values should match the reference
    sv_dev = np.abs(sv - sv_ref).max()
    ok = rel < REL_TOL and sv_dev < 0.05 and sv.max() < SV_MAX
    print(f"[{'PASS' if ok else 'FAIL'}] {name:6s} {str(shape):12s} "
          f"rel_err={rel:.4f}  sv in [{sv.min():.3f}, {sv.max():.3f}]  "
          f"max|sv-sv_ref|={sv_dev:.4f}")
    return ok


def test_torch():
    from muon_torch import muon_ns_fused, muon_ns_reference
    ok = True
    for shape in SHAPES:
        torch.manual_seed(0)
        G = torch.randn(*shape, device="cuda")
        f = muon_ns_fused(G, 5).cpu().numpy()
        r = muon_ns_reference(G, 5).cpu().numpy()
        ok &= report("torch", shape, f, r)
    return ok


def test_jax():
    import jax
    import jax.numpy as jnp
    from muon_jax import muon_ns_fused, muon_ns_reference
    ok = True
    for shape in SHAPES:
        G = jax.random.normal(jax.random.PRNGKey(0), shape)
        f = jax.jit(lambda g: muon_ns_fused(g, 5))(G)
        r = muon_ns_reference(G, 5)
        ok &= report("jax", shape, np.asarray(f), np.asarray(r))
    return ok


def test_cross():
    import jax.numpy as jnp
    from muon_jax import muon_ns_fused as jf
    from muon_torch import muon_ns_fused as tf
    G = np.random.RandomState(0).randn(512, 1024).astype(np.float32)
    a = tf(torch.from_numpy(G).cuda(), 5).cpu().numpy()
    b = np.asarray(jf(jnp.asarray(G), 5))
    diff = np.linalg.norm(a - b) / np.linalg.norm(a)
    print(f"[{'PASS' if diff < 1e-3 else 'FAIL'}] torch-vs-jax fused rel diff = {diff:.2e}")
    return diff < 1e-3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-jax", action="store_true")
    args = parser.parse_args()
    ok = test_torch()
    if not args.skip_jax:
        ok &= test_jax()
        ok &= test_cross()
    print("All Tests Passed" if ok else "Some Tests Failed")
