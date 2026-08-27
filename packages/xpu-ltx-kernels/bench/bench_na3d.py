import sys
import time
from pathlib import Path

import torch

LIB = Path("/home/acm/paul_arc/test/ltx-2-xpu/packages/xpu-ltx-kernels/src/xpu_ltx_kernels/libxpu_ltx_kernels.so")
sys.path.insert(0, "/home/acm/paul_arc/test/ltx-2-xpu/packages/ltx-core/src")
from ltx_core.model.video_vae.transformer.fallback_na.eager import na3d as eager_na3d

torch.ops.load_library(str(LIB))


def bench(name, fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"{name:<22} {dt*1e3:9.3f} ms")
    return dt


def run(shape, kt, kh, kw, reps=20):
    g = torch.Generator(device="xpu").manual_seed(0)
    q = torch.randn(shape, generator=g, dtype=torch.bfloat16, device="xpu")
    k = torch.randn(shape, generator=g, dtype=torch.bfloat16, device="xpu")
    v = torch.randn(shape, generator=g, dtype=torch.bfloat16, device="xpu")

    # eager on XPU (this is what the pipeline runs today)
    te = bench("eager (xpu)", lambda: eager_na3d(q, k, v, kernel_size=[kt, kh, kw], is_causal=None, scale=1.0), reps)
    ts = bench("sycl na3d", lambda: torch.ops.xpu_ltx_kernels.na3d(q, k, v, kt, kh, kw), reps)
    print(f"  -> {te/ts:.1f}x speedup" if ts < te else f"  -> {ts/te:.1f}x SLOWER")


if __name__ == "__main__":
    print("=== typical DiffVAE decode tile, HD=64 ===")
    run((1, 4, 16, 16, 16, 64), 3, 3, 3)
    run((1, 4, 16, 16, 8, 64), 3, 3, 3)
    print("=== larger tile, HD=64 ===")
    run((1, 4, 32, 32, 16, 64), 3, 3, 3)
    print("=== HD=32 ===")
    run((1, 4, 16, 16, 16, 32), 3, 3, 3)