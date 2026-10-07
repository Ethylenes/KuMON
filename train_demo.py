import argparse

import numpy as np

D_IN, D_H, N_CLS = 256, 512, 10
N_TRAIN, BATCH = 20000, 256


def make_data(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(N_TRAIN, D_IN).astype(np.float32)

    # Teacher network makes the labels for a learnable, deterministic task.
    Wt1 = rng.randn(D_IN, 128).astype(np.float32) / np.sqrt(D_IN)
    Wt2 = rng.randn(128, N_CLS).astype(np.float32) / np.sqrt(128)
    y = np.argmax(np.tanh(X @ Wt1) @ Wt2, axis=1).astype(np.int64)
    return X, y


def make_init(seed=1):
    rng = np.random.RandomState(seed)
    def w(o, i): return (rng.randn(o, i) * np.sqrt(2.0 / i)).astype(np.float32)
    return {
        "w0": w(D_H, D_IN), "b0": np.zeros(D_H, np.float32),
        "w1": w(D_H, D_H),  "b1": np.zeros(D_H, np.float32),
        "w2": w(N_CLS, D_H), "b2": np.zeros(N_CLS, np.float32)
    }


def batches(X, y, steps, seed=2):
    rng = np.random.RandomState(seed)
    for _ in range(steps):
        idx = rng.randint(0, len(X), BATCH)
        yield X[idx], y[idx]


def run_torch(args):
    import torch
    import torch.nn.functional as F
    from muon_torch import Muon
    X, y = make_data()
    init = make_init()
    P = {k: torch.nn.Parameter(torch.from_numpy(v).cuda()) for k, v in init.items()}

    def fwd(x):
        h = F.relu(x @ P["w0"].T + P["b0"])
        h = F.relu(h @ P["w1"].T + P["b1"])
        return h @ P["w2"].T + P["b2"]

    muon_opt = Muon([P["w0"], P["w1"]], lr=args.lr, impl=args.impl)
    adam_opt = torch.optim.AdamW([P["b0"], P["b1"], P["w2"], P["b2"]], lr=args.adam_lr)
    losses = []
    for step, (xb, yb) in enumerate(batches(X, y, args.steps)):
        xb, yb = torch.from_numpy(xb).cuda(), torch.from_numpy(yb).cuda()
        loss = F.cross_entropy(fwd(xb), yb)
        muon_opt.zero_grad(); adam_opt.zero_grad()
        loss.backward()
        muon_opt.step(); adam_opt.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"step {step:4d}  loss {losses[-1]:.4f}")
    return losses


def run_jax(args):
    import jax
    import jax.numpy as jnp
    import optax
    import muon_jax
    X, y = make_data()
    params = {k: jnp.asarray(v) for k, v in make_init().items()}
    ns_fn = muon_jax.muon_ns_fused if args.impl == "fused" else muon_jax.muon_ns_reference
    label_fn = lambda p: {k: ("muon" if k in ("w0", "w1") else "adam") for k in p}
    tx = muon_jax.muon(args.lr, adam_learning_rate=args.adam_lr, label_fn=label_fn, ns_fn=ns_fn)
    opt_state = tx.init(params)

    def loss_fn(p, x, yb):
        h = jax.nn.relu(x @ p["w0"].T + p["b0"])
        h = jax.nn.relu(h @ p["w1"].T + p["b1"])
        logits = h @ p["w2"].T + p["b2"]
        return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()

    @jax.jit
    def step_fn(p, s, x, yb):
        loss, grads = jax.value_and_grad(loss_fn)(p, x, yb)
        updates, s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    losses = []
    for step, (xb, yb) in enumerate(batches(X, y, args.steps)):
        params, opt_state, loss = step_fn(params, opt_state, jnp.asarray(xb), jnp.asarray(yb))
        losses.append(float(loss))
        if step % 50 == 0:
            print(f"step {step:4d}  loss {losses[-1]:.4f}")
    return losses


if __name__ == "__main__":
    """Loss curves are saved to 'curves_<jax/torch>_<fused/reference>' path"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=["torch", "jax"], required=True)
    parser.add_argument("--impl", choices=["fused", "reference"], default="fused")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--adam_lr", type=float, default=1e-3)
    args = parser.parse_args()

    losses = run_torch(args) if args.framework == "torch" else run_jax(args)
    np.save(f"curves_{args.framework}_{args.impl}.npy", np.array(losses))
    print("saved", f"curves_{args.framework}_{args.impl}.npy")