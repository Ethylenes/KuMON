import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")  # T4 = Turing

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="muon_ns_ext",
            sources=[os.path.join(HERE, "muon_ns.cu"), os.path.join(HERE, "muon_torch.cpp")],
            extra_include_paths=[HERE],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            extra_ldflags=["-lcublas"],
            verbose=True,
        )
    return _ext


A_, B_, C_ = 3.4445, -4.7750, 2.0315


@torch.no_grad()
def muon_ns_reference(G: torch.Tensor, steps: int = 5, dtype=torch.float16) -> torch.Tensor:
    """Naive PyTorch Newton-Schulz (correctness oracle + speed baseline)."""
    X = G.float()
    X = X / (X.norm() + 1e-7)
    X = X.to(dtype)
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = B_ * A + C_ * (A @ A)
        X = A_ * X + B @ X
    if tall:
        X = X.T
    return X.float()


@torch.no_grad()
def muon_ns_fused(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Fused CUDA Newton-Schulz. Returns float32 with G's shape."""
    ext = _get_ext()
    X = G.float().contiguous()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T.contiguous()
    m, n = X.shape
    out = torch.empty_like(X)
    ws = torch.empty(ext.workspace_bytes(m, n), dtype=torch.uint8, device=X.device)
    ext.forward(X, out, ws, steps)
    return out.T if tall else out


class Muon(torch.optim.Optimizer):
    """
    Muon for 2D params, AdamW for others.
    impl: "fused" (CUDA kernel) or "reference" (plain PyTorch).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0, impl="fused"):
        assert impl in ("fused", "reference")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.ns = muon_ns_fused if impl == "fused" else muon_ns_reference

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.lerp_(g, 1 - mom)                       # EMA momentum
                upd = g.lerp(buf, mom) if group["nesterov"] else buf
                rows = p.size(0)
                cols = p.numel() // rows
                o = self.ns(upd.reshape(rows, cols), group["ns_steps"])
                o = o.reshape(p.shape) * max(1.0, rows / cols) ** 0.5
                if group["weight_decay"] != 0:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(o.to(p.dtype), alpha=-lr)
        return loss