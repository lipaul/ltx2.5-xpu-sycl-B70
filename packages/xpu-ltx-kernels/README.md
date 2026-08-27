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

`ltx-core`'s DiffVAE NA fallback picks up the kernel automatically via
`xpu_ltx_kernels.available()` (see `fallback_na/__init__.py`):
natten → `xpu_ltx_kernels` (XPU) → Triton (CUDA) → eager tiled SDPA. No manual
wiring needed; the loud no-natten banner logs `xpu_ltx_kernels SYCL na3d`.

## Kernel contract

`na3d(q, k, v, kt, kh, kw) -> out`, all `(B, T, H, W, NH, HD)` bf16 contiguous
on XPU, `HD` in {32, 64}. Non-causal windows: per-axis
`start = min(max(i - k/2, 0), L - k)`, `end = start + k`. `scale` is 1.0 (Q is
pre-scaled upstream). fp32 accumulate, flash-style online softmax, bf16 output.
Semantics match `ltx_core...fallback_na.eager.na3d`; see
`tests/test_na3d_parity.py`.

## ESIMD note

An ESIMD port was attempted (`esimd::simd` register tiles, `sycl_explicit_simd`
kernels, `-fsycl-esimd-force-stateless-mem`). The device-code JIT on this
Intel Graphics driver rejects ESIMD-generated SPIR-V with
`Build program log ... invalid api option` even for a trivial `simd<float,16>`
copy kernel, so the production path is plain SYCL (the compiler auto-vectorizes
the bf16 loads). AOT (`-fsycl-targets=spir64_gen --device bmg`) fails the
ESIMD kernel build (`gen compiler command failed`). Revisit with a newer
oneAPI/driver.

## Benchmark

`bench/bench_na3d.py` compares the raw op vs the eager tiled-SDPA fallback on
XPU (2–8x faster, larger tiles win more). `bench/bench_diffvae_block.py`
compares a full `NeighborhoodAttention3D.forward` (RoPE + QKV + proj included):
~1.2x at tile sizes 16×16, where the NA is not the sole cost.