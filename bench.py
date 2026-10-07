import argparse
import time 

import torch


SHAPES = [(512, 512), (1024, 1024), (768, 3072), (2048, 2048), (2048, 8192)]
WARMUP, ITERS = 10, 50

def time_torch(fn, G):
    for _ in range(WARMUP):
        fn(G, 5)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn(G, 5)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def time_jax(fn, G):
    out = fn(G)
    out.block_until_ready()
    for _ in range(WARMUP):
        fn(G).block_until_ready()
    t = time.perf_counter()
    for _ in range(ITERS):
        out = fn(G)
    out.block_until_ready()
    return (time.perf_counter() - t) / ITERS * 1e3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-jax", action="store_true")
    args = parser.parse_args()

    from muon_torch import muon_ns_fused, muon_ns_reference
    if not args.skip_jax:
        import jax 
        from muon_jax import muon_ns_fused as jfused, muon_ns_reference as jref
        jref_j = jax.jit(lambda g: jref(g, 5))
        jfused_j = jax.jit(lambda g: jfused(g, 5))


    print(f"{'shape':>14} | {'torch ref':>10} {'torch fused':>12} {'speedup':>8}" + ("" if args.skip_jax else f" | {'jax ref':>9} {'jax fused':>10} {'speedup':>8}"))
    for shape in SHAPES:
        G = torch.randn(*shape, device="cuda")
        tref = time_torch(muon_ns_reference, G)
        tfused = time_torch(muon_ns_fused, G)
        output = f"{str(shape):>14} | {tref:9.3f}ms {tfused:10.3f}ms {tref / tfused:7.2f}x"
        if not args.skip_jax:
            import jax.numpy as jnp
            Gj = jnp.asarray(G.cpu().numpy())
            jr = time_jax(jref_j, Gj)
            jf = time_jax(jfused_j, Gj)
            output += f" | {jr:8.3f}ms {jf:9.3f}ms {jr / jf:7.2f}x"
        print(output)

