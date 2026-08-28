import sys
from pathlib import Path

import torch

LIB = Path("/home/acm/paul_arc/test/ltx-2-xpu/packages/xpu-ltx-kernels/src/xpu_ltx_kernels/libxpu_ltx_kernels.so")

sys.path.insert(0, "/home/acm/paul_arc/test/ltx-2-xpu/packages/ltx-core/src")
from ltx_core.model.video_vae.transformer.fallback_na.eager import na3d as eager_na3d

torch.ops.load_library(str(LIB))


def make_inputs(b, t, h, w, nh, hd, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (b, t, h, w, nh, hd)
    q = torch.randn(shape, generator=g, dtype=torch.bfloat16)
    k = torch.randn(shape, generator=g, dtype=torch.bfloat16)
    v = torch.randn(shape, generator=g, dtype=torch.bfloat16)
    return q, k, v


def run_case(b, t, h, w, nh, hd, kt, kh, kw, seed=0, op="na3d"):
    q, k, v = make_inputs(b, t, h, w, nh, hd, seed)
    qx, kx, vx = q.to("xpu"), k.to("xpu"), v.to("xpu")
    ref = eager_na3d(q, k, v, kernel_size=[kt, kh, kw], is_causal=None, scale=1.0)
    got = getattr(torch.ops.xpu_ltx_kernels, op)(qx, kx, vx, kt, kh, kw).cpu()
    torch.xpu.synchronize()
    diff = (got.float() - ref.float()).abs().max().item()
    rel = diff / (ref.float().abs().max().item() + 1e-6)
    ok = torch.allclose(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
    print(f"[{op}] B{b} T{t} H{h} W{w} NH{nh} HD{hd} k=({kt},{kh},{kw}) seed{seed}: "
          f"ok={ok} max_abs_diff={diff:.4f} rel={rel:.4f}")
    return ok


if __name__ == "__main__":
    cases = [
        # typical DiffVAE decode tiles: kernel (3,3,3), HD 64/32
        (1, 4, 8, 8, 16, 64, 3, 3, 3),
        (1, 4, 8, 8, 16, 64, 3, 3, 3, 1),
        (1, 4, 8, 8, 16, 32, 3, 3, 3),
        (1, 4, 16, 16, 8, 64, 3, 3, 3),
        (2, 4, 8, 8, 16, 64, 3, 3, 3),
        # odd W + non-cubic kernel
        (1, 4, 8, 7, 16, 64, 3, 3, 3),
        (1, 4, 8, 8, 16, 64, 1, 1, 1),
        (1, 6, 8, 8, 16, 64, 5, 3, 3),
        (1, 4, 8, 8, 16, 64, 3, 5, 3),
        # borders: T/H/W == kernel
        (1, 3, 3, 3, 8, 64, 3, 3, 3),
        # small dims (>= kernel, matching NeighborhoodAttention3D.forward guard)
        (1, 3, 3, 3, 4, 64, 3, 3, 3),
        (1, 5, 3, 4, 4, 64, 3, 3, 3),
        # real decoder kernels (HD=64 only; na3d_dpas is HD-64-only)
        (1, 9, 16, 16, 8, 64, 3, 7, 7),
        (1, 9, 16, 16, 8, 64, 3, 5, 5),
        (1, 11, 16, 16, 8, 64, 11, 11, 11),
        (1, 9, 32, 16, 4, 64, 3, 7, 7),
    ]
    ok_all = True
    for case in cases:
        ok_all &= run_case(*case, op="na3d")
        _, t, h, w, _, hd, kt, kh, kw = case[:9]
        if hd == 64 and t >= kt and h >= kh and w >= kw:  # na3d_dpas needs dims>=kernel + HD=64
            ok_all &= run_case(*case, op="na3d_dpas")
            ok_all &= run_case(*case, op="na3d_dpas2")
    print("\nALL OK" if ok_all else "\nFAILURES")
    sys.exit(0 if ok_all else 1)