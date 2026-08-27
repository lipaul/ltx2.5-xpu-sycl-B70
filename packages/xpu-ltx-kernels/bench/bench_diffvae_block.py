import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/acm/paul_arc/test/ltx-2-xpu/packages/ltx-core/src")
from ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D
from ltx_core.model.video_vae.transformer.fallback_na import (
    EagerSdpaAttention,
    XpuNaAttention,
)

LIB = Path("/home/acm/paul_arc/test/ltx-2-xpu/packages/xpu-ltx-kernels/src/xpu_ltx_kernels/libxpu_ltx_kernels.so")
torch.ops.load_library(str(LIB))


def bench(fn, n=15, warmup=3):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n


def run(dim, t, h, w, kt, kh, kw, head_dim=64, batch=1):
    torch.manual_seed(0)
    x = torch.randn(batch, t, h, w, dim, dtype=torch.bfloat16, device="xpu")
    attn = NeighborhoodAttention3D(dim, (kt, kh, kw), head_dim=head_dim).to("xpu")
    attn = attn.to(torch.bfloat16)

    attn.attention_function = EagerSdpaAttention()
    out_ref = attn(x)
    torch.xpu.synchronize()

    attn.attention_function = XpuNaAttention()
    out_sycl = attn(x)
    torch.xpu.synchronize()

    ok = torch.allclose(out_ref.float(), out_sycl.float(), rtol=2e-2, atol=2e-2)

    # measure via forward with each backend
    def run_eager():
        attn.attention_function = EagerSdpaAttention()
        attn(x)

    def run_sycl():
        attn.attention_function = XpuNaAttention()
        attn(x)

    te = bench(run_eager)
    ts = bench(run_sycl)
    print(f"dim={dim} T{t} H{h} W{w} k=({kt},{kh},{kw}) hd={head_dim}: "
          f"parity={ok} eager={te*1e3:.2f}ms sycl={ts*1e3:.2f}ms -> {te/ts:.1f}x")
    return ok


if __name__ == "__main__":
    # DiffVAE decode tiles at 121f/1536x1024 → latent spatial 48x32, tile ~(4,16,16)
    all_ok = True
    all_ok &= run(1024, 4, 16, 16, 3, 3, 3, head_dim=64)
    all_ok &= run(1024, 4, 16, 16, 3, 3, 3, head_dim=32)
    all_ok &= run(2048, 4, 16, 16, 3, 3, 3, head_dim=64)
    all_ok &= run(512, 4, 16, 16, 3, 3, 3, head_dim=64)
    print("\nALL OK" if all_ok else "\nFAILURES")