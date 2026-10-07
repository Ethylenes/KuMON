import ctypes
import os
import shutil
import subprocess
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax


PATH = os.path.dirname(os.path.abspath(__file__))
_ffi = jax.ffi if hasattr(jax, "ffi") else __import__("jax.extend.ffi", fromlist=["x"])
_registered = False

def build(force: bool = False) -> str:
    so = os.path.join(PATH, "libmuon_jax.so")
    srcs = [os.path.join(PATH, f) for f in ("muon_ns.cu", "muon_jax.cu", "muon_ns.h")]
    if (not force and os.path.exists(so) and os.path.getmtime(so) > max(os.path.getmtime(s) for s in srcs)):
        return so
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cmd = [nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
           "-w", "-Xcompiler", "-w",   # silence warnings from XLA's headers
           "-gencode=arch=compute_75,code=sm_75",       # T4
           f"-I{_ffi.include_dir()}", f"-I{PATH}",
           os.path.join(PATH, "muon_ns.cu"), os.path.join(PATH, "muon_jax.cu"),
           "-lcublas", "-o", so]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    return so

def _register():
    global _registered
    if _registered:
        return
    lib = ctypes.cdll.LoadLibrary(build())
    _ffi.register_ffi_target("muon_ns", _ffi.pycapsule(lib.MuonNs), platform="CUDA")
    _registered = True


A_, B_, C_ = 3.4445, -4.7750, 2.0315

def muon_ns_reference(G, steps: int = 5, dtype=jnp.float16):
    X = G.astype(jnp.float32)
    X = (X / (jnp.linalg.norm(X) + 1e-7)).astype(dtype)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = B_ * A + C_ * (A @ A)
        X = A_ * X + B @ X
    if tall:
        X = X.T
    return X.astype(jnp.float32)

def muon_ns_fused(G, steps: int = 5):
    """
    Fused CUDA NS via XLA FFI
    float32 in/out
    jit compatible
    """
    _register()
    X = G.astype(jnp.float32)
    is_tall = X.shape[0] > X.shape[1]
    if is_tall:
        X = X.T
    call = _ffi.ffi_call("muon_ns",
                         jax.ShapeDtypeStruct(X.shape, jnp.float32),
                         vmap_method="sequential")
    out = call(X, steps=np.int64(steps))
    return out.T if is_tall else out

# OPTAX
class MuonState(NamedTuple):
    momentum: Any

def scale_by_muon(
    momentum: float = 0.95, nesterov: bool = True, 
    ns_steps: int = 5, ns_fn: Callable = muon_ns_fused
) -> optax.GradientTransformation:
    """
    Momentum + NS ortho (& shape scaling)
    Non-2D leaves are passed throuhg unchanged (uses AdamW, see the muon function)
    """
    def init_fn(params):
        return MuonState(jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):
        del params
        mu = jax.tree_util.tree_map(
            lambda g, m : momentum * m + (1 - momentum) * g, updates, state.momentum
        )
        if nesterov:
            eff = jax.tree_util.tree_map(
                lambda g, m : (1 - momentum) * g + momentum * m, updates, mu
            )
        else:
            eff = mu

        def orth(u):
            if u.ndim != 2:
                return u
            o = ns_fn(u, ns_steps)
            r, c = u.shape
            return (o * max(1.0, r / c) ** 0.5).astype(u.dtype)

        return jax.tree_util.tree_map(orth, eff), MuonState(mu)
    return optax.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    weight_decay: float = 0.0,
    adam_learning_rate: float = 1e-3,
    adam_weight_decay: float = 0.0,
    label_fn: Optional[Callable] = None,
    ns_fn: Callable = muon_ns_fused
) -> optax.GradientTransformation:
    if label_fn is None:
        label_fn = lambda params : jax.tree_util.tree_map(
            lambda p : "muon" if p.ndim == 2 else "adam", params
        )
    muon_tx = optax.chain(
        scale_by_muon(momentum, nesterov, ns_steps, ns_fn),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate)
    )
    adam_tx = optax.adamw(adam_learning_rate, weight_decay=adam_weight_decay)
    return optax.multi_transform({"muon": muon_tx, "adam": adam_tx}, label_fn)
