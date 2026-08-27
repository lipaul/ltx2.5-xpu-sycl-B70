# xpu-ltx-kernels

SYCL kernels for `ltx-core` on Intel XPU (B70). Currently ships one fused kernel:

- **`na3d`** — 3D neighborhood attention for the DiffVAE decoder, the SYCL port
  of `ltx_kernels`' Blackwell-only `na_attn_dsl`. On XPU the decoder otherwise
  runs the slow pure-PyTorch eager tiled-SDPA fallback (natten is CUDA-only and
  the Triton fallback is gated on `torch.cuda.is_available()`).

This package is excluded from the uv workspace (like `ltx-kernels`). Build the
native library with CMake, then install the Python wrapper.

## Build

Prerequisites:

- the **oneAPI 2025.3** compiler (`intel-oneapi-compiler-dpcpp-cpp-2025.3` via
  the Intel apt repo). This is a hard requirement: torch 2.12.0+xpu links
  `libsycl.so.8` (2025.3), and the newer 2026.x compilers ship `libsycl.so.9`
  whose device-image format is incompatible (crashes in
  `ProgramManager::addImage` at dlopen). CMake picks 2025.3 automatically when
  present; verify with:
  ```bash
  apt-cache policy intel-oneapi-compiler-dpcpp-cpp-2025.3   # installable?
  sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp-2025.3
  /opt/intel/oneapi/compiler/2025.3/bin/icpx --version
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
flash kernel that can't use XMX/DPAS (ESIMD is blocked by the driver). Measured
at decode-tile shapes:

| shape              | kernel      | sycl / eager |
|--------------------|-------------|--------------|
| (1,9,16,16,32,64)  | (3,7,7)     | 0.4x         |
| (1,9,16,16,32,64)  | (3,5,5)     | 1.0x         |
| (1,9,32,32,32,64)  | (3,7,7)     | 0.9x         |
| (1,9,16,32,32,64)  | (3,7,7)     | 1.2x         |
| (1,11,32,32,32,64) | (11,11,11)  | 0.2x         |

End-to-end (512×512/49f, distilled, `--offload cpu`): **eager decode 6.4s vs
SYCL 17.9s** (~2.8x slower). So the SYCL kernel is a correctness parity port,
not a speedup, for the shipped decoder — keep eager as the XPU default. A
real win would need XMX/DPAS (blocked) or a batched-GEMM-shaped kernel, not
per-query flash.

`bench/bench_diffvae_block.py` compares a full `NeighborhoodAttention3D.forward`
(RoPE + QKV + proj included) at cubic kernels: ~1.2x, where the NA is not the
sole cost.