# xpu-ltx-kernels

SYCL kernels for `ltx-core` on Intel XPU (B70). Currently ships:

- **`na3d`** — 3D neighborhood attention for the DiffVAE decoder, the SYCL port
  of `ltx_kernels`' Blackwell-only `na_attn_dsl`. On XPU the decoder otherwise
  runs the pure-PyTorch eager tiled-SDPA fallback (natten is CUDA-only and the
  Triton fallback is gated on `torch.cuda.is_available()`).
- **`na3d_esimd`** — same kernel written with Intel Explicit SIMD (ESIMD).
  Requires the oneAPI 2026.x toolchain / torch 2.13.0+xpu (libsycl.so.9); with
  the older 2025.3 pairing the ESIMD device code was driver-rejected.
- **`na3d_dpas`** — batched-GEMM-shaped NA via ESIMD **DPAS** (XMX systolic),
  with VNNI-packed operands. Correctness-parity vs eager; performance work is
  ongoing (see Benchmark).
- **`na3d_dpas2`** — tuned `na3d_dpas`: window-union halo, hoisted per-chunk
  key decode, MQ=8 online softmax. Correct everywhere; 0.9x eager on the
  stage-1 (3,7,7) kernel at NH=8 (see Benchmark).

This package is excluded from the uv workspace (like `ltx-kernels`). Build the
native library with CMake, then install the Python wrapper.

## Build

Prerequisites:

- the **oneAPI 2026.1** compiler (`intel-oneapi-compiler-dpcpp-cpp-2026.1` via
  the Intel apt repo). This must match torch-xpu's SYCL runtime: torch
  2.13.0+xpu links `libsycl.so.9` (oneAPI 2026.0), and the 2026.1 compiler's
  device code is accepted by the driver. Do NOT use the 2025.3 toolchain
  (libsycl.so.8): its ESIMD device code is rejected with `invalid api option`
  (malformed `-vc-codegen` option token). CMake picks 2026.1 automatically when
  present; verify with:
  ```bash
  apt-cache policy intel-oneapi-compiler-dpcpp-cpp-2026.1   # installable?
  sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp-2026.1
  /opt/intel/oneapi/compiler/2026.1/bin/icpx --version
  ```
- the project venv with torch-xpu (`uv sync` at the repo root).

Build:

```bash
cmake -S packages/xpu-ltx-kernels -B packages/xpu-ltx-kernels/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=$PWD/.venv/bin/python
cmake --build packages/xpu-ltx-kernels/build -j
uv pip install -e packages/xpu-ltx-kernels --no-build-isolation
```

The `.so` lands in `src/xpu_ltx_kernels/`; the editable install makes
`import xpu_ltx_kernels` load it. Re-run `cmake` (not just `cmake --build`)
after adding/removing `csrc/*.cpp` — the source list is a configure-time GLOB.

## Integration

`ltx-core`'s DiffVAE NA fallback chain is
natten → Triton (CUDA) → eager tiled SDPA. The SYCL kernel is **opt-in** on XPU:
set `LTX_XPU_NA_KERNELS=1` to select it via `fallback_na_attention()`. It is not
the default because end-to-end it is *slower* than eager on the real decoder
kernels (see Benchmark).

## Kernel contract

`na3d(q, k, v, kt, kh, kw) -> out`, all `(B, T, H, W, NH, HD)` bf16 contiguous
on XPU, `HD` in {32, 64}. Non-causal windows: per-axis
`start = min(max(i - k/2, 0), L - k)`, `end = start + k`. `scale` is 1.0 (Q is
pre-scaled upstream). fp32 accumulate, flash-style online softmax, bf16 output.
Semantics match `ltx_core...fallback_na.eager.na3d`; see
`tests/test_na3d_parity.py`.

## Benchmark (measured, end-to-end)

`bench/bench_na3d.py` compares the raw op vs the eager tiled-SDPA fallback on
XPU for cubic `(3,3,3)` kernels — SYCL is 2–8x faster there because eager's
mask materialization dominates small kernels.

But the **real DiffVAE decoder uses wide kernels**: `stage_kernels =
[[3,7,7],[3,7,7],[3,5,5],[3,5,5],[11,11,11]]`. At those, eager's dense batched
SDPA (one big masked GEMM per window-geometry group) is faster than a per-query
flash kernel. Measured on torch 2.13 at decode-tile shapes:

| shape              | kernel      | sycl / eager | esimd / eager |
|--------------------|-------------|--------------|---------------|
| (1,9,16,16,32,64)  | (3,7,7)     | 0.5x         | 0.1x          |
| (1,9,16,16,32,64)  | (3,5,5)     | 1.0x         | 0.2x          |
| (1,9,32,32,32,64)  | (3,7,7)     | 0.7x         | 0.2x          |
| (1,9,16,32,32,64)  | (3,7,7)     | 1.0x         | 0.2x          |
| (1,11,16,16,32,64) | (11,11,11)  | 0.1x         | 0.02x         |

`na3d_dpas2` (window-union halo + hoisted key decode + online softmax, MQ=8)
is a tuned variant: correct everywhere and **0.9x eager at the stage-1 real
kernel (3,7,7) 9x16x16** (48ms vs eager 42ms, measured NH=8). It does not yet
beat eager at NH=32 or on the small-window kernels (3,5,5)/(11,11,11), where
eager's whole-volume batched SDPA is dramatically faster (per-query DPAS
cannot match eager's batching because DPAS M<=8).

`na3d_dpas` (DPAS XMX, batched QK^T/PV with VNNI operands) is **correct**
(full parity vs eager at `rtol/atol 2e-2` on all real kernels) but, in its
current fine-grained grid, slower than eager (e.g. (3,7,7) 9×16×16: dpas 64ms
vs eager 42ms) — too many small work-items, each recomputing the full halo
QK^T for only 8 queries (halo 560 keys vs a 147-key window ≈ 4x wasted DPAS),
and the per-query softmax on the critical path. Beating eager needs a
persistent grid with M=64-style query tiles and the DPAS PV off the softmax
critical path (as `na_attn_dsl`'s cooperative tiles do). Status: work in
progress; eager stays the default.

End-to-end (512×512/49f, distilled, `--offload cpu`): **eager decode 7.0s vs
SYCL 17.3s** (~2.5x slower). So the per-query kernels are correctness-parity
ports, not speedups, for the shipped decoder — keep eager as the XPU default.
A real win would need a batched-GEMM-shaped (DPAS/XMX) kernel that matches
eager's dense-GEMM structure; per-query flash (plain SYCL or ESIMD) cannot beat
it. `na3d_esimd` is kept as proof that ESIMD runs on this stack since the
torch 2.13 / oneAPI 2026.1 upgrade.

`bench/bench_diffvae_block.py` compares a full `NeighborhoodAttention3D.forward`
(RoPE + QKV + proj included) at cubic kernels: ~1.2x, where the NA is not the
sole cost.