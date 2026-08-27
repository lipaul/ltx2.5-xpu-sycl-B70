# Porting `ltx-kernels` to Intel XPU (B70): SYCL `na3d` for the DiffVAE decoder

This file documents the end-to-end process of porting a CUDA kernel from
`packages/ltx-kernels` to Intel XPU in a new package, `packages/xpu-ltx-kernels`.
It is a record of decisions, dead ends, and hard-won facts so future porting work
does not repeat the same investigation. The companion living docs are
`packages/xpu-ltx-kernels/README.md` (build/usage) and `AGENTS.md` (toolchain
rules).

Final result: a plain-SYCL `na3d` (3D neighborhood attention) kernel that
replaces the eager tiled-SDPA fallback in DiffVAE decode, **2–8× faster for the
raw op** (up to 5× at typical 16×16 decode tiles, 8.4× for HD=32), ~1.2× for a
full `NeighborhoodAttention3D.forward` where RoPE/linear dominate. Committed as
`b316844`.

---

## 1. Why this kernel (target selection)

The user asked for "one XPU kernel in `xpu-ltx-kernels`, mimicking the CUDA
optimizations, to accelerate video inference." We surveyed `ltx-kernels`
(two compiled extensions + CuTe DSL VAE kernels):

| Kernel family         | Relevance on XPU                                         |
|-----------------------|-----------------------------------------------------------|
| `all2all_cpp`         | Multi-GPU tensor-parallel comms; single-GPU XPU → skip    |
| `ops_cpp` (rms_norm_rope, FP6 pack) | Only pays off with a blockwise FP8 GEMM path (not on XPU) |
| `blockwise_cpp` (FP8 GEMM) | Biggest compute win if BMG FP8 XMX works; by far the largest porting effort, highest risk |
| `nvfp4_cpp`           | Blackwell-only (SM≥10); no FP4 tensor cores on BMG → skip |
| `vae` CuTe DSL (`na_attn_dsl` / `block_fna_dsl`) | The real hot spot on XPU — see below |

**The winning target: 3D neighborhood attention in the DiffVAE decoder.** The
inference entrypoint (`run_pipeline.sh`) uses the diffusion video VAE
(`ltx-2.5-video-vae-bf16.safetensors`), whose decoder runs many `NABlock` /
`DiffusionNABlock` blocks. Its NA backend chain is:

```
natten → Triton → eager tiled SDPA
```

On XPU this is broken: `natten` is CUDA-only (dropped from this fork), and
`triton_na_available()` in `fallback_na/__init__.py` is gated on
`torch.cuda.is_available()`. So **every default inference run pays the pure
PyTorch eager tiled-SDPA fallback** (`fallback_na/eager.py` — stack/permute +
full additive-mask materialization). That is a clean, self-contained porting
target with a reference implementation already in-tree for correctness testing.

## 2. Kernel contract (fixed)

`na3d(q, k, v, kt, kh, kw) -> out`, all `(B, T, H, W, NH, HD)` bf16 contiguous
on XPU. Non-causal NATTEN semantics (from `eager.na3d`):

```
per-axis window:  start = min(max(i - k/2, 0), L - k),  end = start + k
```

`scale` is 1.0 at the call site (Q is pre-scaled by `head_dim**-0.5` upstream in
`NeighborhoodAttention3D.forward`). RoPE/QKV/proj stay in PyTorch; the kernel
does only windowed attention. `HD` in {32, 64}; spatial dims must be ≥ kernel
(matching the upstream `forward` guard).

## 3. The toolchain minefield (the long tail of this port)

This consumed most of the effort and is the part that will bite any future
SYCL-on-this-stack work. Facts, in the order we learned them:

### 3.1 torch-xpu links `libsycl.so.8` — the compiler must match it

`torch 2.12.0+xpu` links `libsycl.so.8` (oneAPI **2025.3**). The standalone
compilers installed at the time were 2026.0/2026.1, which ship `libsycl.so.9`.
Consequences of building with a v9 compiler:

- **v9 headers + v9 runtime**: compiles and runs standalone, but loading the
  `.so` into the torch process crashes — two SYCL runtimes in one process,
  `UR_RESULT_ERROR_UNINITIALIZED` at static init. (Preloading v9 via ctypes
  crashes torch itself.)
- **v9 headers + v8 runtime**: link fails, `submit_with_event_impl` /
  `submit_kernel_direct_with_event_impl` undefined (v9-only runtime internals).
- **v8 headers + v9 compiler**: `stl_wrappers/cmath` breaks — the v8
  `defines_elementary.hpp` lacks the `__DPCPP_SYCL_EXTERNAL_LIBC` fallback the
  v9 compiler emits. A `-D__DPCPP_SYCL_EXTERNAL_LIBC=__DPCPP_SYCL_EXTERNAL`
  workaround compiles and links libsycl.so.8, but the v9-generated **device
  image format** is incompatible with libsycl.so.8: dlopen crashes in
  `ProgramManager::addImage`.
- **venv's bundled SYCL (v8 headers + libsycl.so.8) is self-consistent**, but
  the venv ships no compiler.

**Fix:** install the matching compiler from the Intel apt repo
(`/etc/apt/sources.list.d/oneAPI.list`):
`intel-oneapi-compiler-dpcpp-cpp-2025.3` (2025.3.3-30). Its icpx + headers +
`libsycl.so.8` all match torch. CMake selects `compiler/2025.3` when present.
`torch.utils.cpp_extension.SyclExtension` resolves `SYCL_HOME` from `icpx` on
PATH — with 2025.3 first that also works, but CMake gives us `TORCH_LIBRARY`
control, so we use CMake.

### 3.2 ESIMD is rejected by the current driver

The user explicitly asked for ESIMD/SYCL. We wrote a proper ESIMD kernel
(`esimd::simd` register tiles, `__attribute__((sycl_explicit_simd))`,
`-fsycl-esimd-force-stateless-mem`). Even a trivial `simd<float,16>` copy kernel
fails at JIT:

```
Build program log for 'Intel(R) Graphics [0xe223]': invalid api option: ...
```

AOT (`-fsycl-targets=spir64_gen --device bmg` / `-Xs "-device bmg"`) fails with
`gen compiler command failed` / "No kernel named ... found". **The driver rejects
ESIMD-generated SPIR-V on this stack.** Verdict: plain SYCL is the working path;
the compiler auto-vectorizes bf16 loads. Documented in README + AGENTS.md; the
ESIMD source was removed from the build. Revisit only with a newer
oneAPI/driver.

### 3.3 bf16 device conversions crash on this stack

`sycl::ext::oneapi::bfloat16` conversions in device code segfault the process.
Workaround: treat bf16 as raw `uint16_t` and bit-twiddle (`(uint32_t)h << 16`
up, `(bits + 0x7fff + ((bits>>16)&1)) >> 16` down, RNE). This is stable across
the stack and matched all parity tests.

### 3.4 SYCL build quirks

- **CMake has no `.sycl` extension support** — it silently drops `.sycl`
  sources and links an empty `.so` (no object files, `xpu_ltx_kernels_OBJECTS`
  empty). Name SYCL sources `.cpp` and compile with icpx `-fsycl`.
- **One `TORCH_LIBRARY` block per namespace** — the second block throws
  `Only a single TORCH_LIBRARY can be used to register the namespace ...` even
  as `TORCH_LIBRARY_FRAGMENT` at dlopen. Put every `m.def` in one block.
- Keep `-I` for the **compiler's** SYCL headers only; do not add the venv's
  `$VENV/include` (v8 headers) to the include path — it shadows `<sycl/...>`
  and breaks v8/v9 mixing. The venv root include also doubles as the Python
  include dir, which is how it leaks in under `setuptools`.
- torch custom-op schema types: use `double` in the C++ signature for a `float`
  schema arg (standard torch rule, not SYCL-specific).

### 3.5 Rebuild gotchas

- Re-run `cmake` (not just `cmake --build`) after adding/removing `csrc/*.cpp`
  — the source list is a configure-time `file(GLOB ...)`.
- The bash tool's persistent cwd broke several `cd build && cmake` invocations;
  always pass an explicit workdir.

## 4. Kernel implementation

`csrc/na3d.cpp` — plain SYCL, one work-item per `(b,t,h,nh)` row × a W position
(`parallel_for(range<2>(W, B*T*H*NH))`):

1. decode the flattened row back to `(b,t,h,nh)`;
2. load the query into a `float[HDP/8][8]` register tile;
3. flash-style online softmax over the `kt*kh*kw` window
   (`m = max(m,s)`, `l = l*α + p`, `acc = acc*α + v*p`, fp32 accumulate);
4. normalize `acc / l` and store bf16.

Template-specialized on `HD` ∈ {32, 64}. Registered via
`TORCH_LIBRARY(xpu_ltx_kernels, m)` → `torch.ops.xpu_ltx_kernels.na3d`.
`sycl::queue` comes from `c10::xpu::getCurrentXPUStream()` (the torch-xpu stream
IS a SYCL queue — converts via `operator sycl::queue&`).

Experiments that did **not** win and were reverted:

- **W-blocked kernel** (WB=2/4 W positions per work-item to reuse shared K/V
  columns): register pressure from `WB × HD` fp32 accumulators spilled and
  regressed vs the per-query kernel on small tiles; the per-query kernel won.
- **ESIMD** (see 3.2).

## 5. Packaging and integration

- `packages/xpu-ltx-kernels/` excluded from the uv workspace (like
  `ltx-kernels`). CMake emits `libxpu_ltx_kernels.so` into
  `src/xpu_ltx_kernels/`; `uv pip install -e packages/xpu-ltx-kernels
  --no-build-isolation` makes `import xpu_ltx_kernels` load it.
- `src/xpu_ltx_kernels/__init__.py`: `load()` (never raises), `available()`
  (XPU present + library loadable + op registered), `na3d()` convenience.
- ltx-core wiring (both in `fallback_na/__init__.py` and `apply.py`):
  `fallback_na_attention()` now returns
  `XpuNaAttention` → `TritonNaAttention` → `EagerSdpaAttention`, and the
  EAGER_SDPA branch of `_install_attention` delegates to
  `fallback_na_attention()` instead of hard-coding eager. So the selection is
  automatic: **natten → xpu_ltx_kernels (XPU) → Triton (CUDA) → eager**.
  The no-natten banner logs `xpu_ltx_kernels SYCL na3d`.

## 6. Verification and benchmarks

- `tests/test_na3d_parity.py`: 12 cases (HD 32/64, B 1–2, odd W, non-cubic
  kernels, border dims == kernel) vs `eager.na3d` on CPU. `max_abs_diff` 0.0156
  (bf16 rounding), all `allclose(rtol=2e-2, atol=2e-2)`, **ALL OK**.
- `bench/bench_na3d.py` (raw op, XPU, eager vs SYCL) — representative numbers:

  | shape (T,H,W,NH,HD)        | eager  | sycl   | speedup |
  |----------------------------|--------|--------|---------|
  | (4,16,16,16,64)            | ~0.4ms | ~0.12ms| ~3×     |
  | (4,32,32,16,64)            | ~2.1ms | ~0.42ms| ~5×     |
  | (4,16,16,16,32)            | ~0.38ms| ~0.046ms| ~8.4×  |

- `bench/bench_diffvae_block.py`: full `NeighborhoodAttention3D.forward` (RoPE
  + QKV + proj included) ~1.2× — linear/RoPE dominate at 16×16 tiles, so the
  raw-op win shrinks; expect more at larger decode tiles.
- `uv run ruff check` clean on all changed files.

## 7. End-to-end run status

Full pipeline A/B (`run_pipeline.sh`) was **not** run: no model weights on this
machine (`/home/acm/paul/models/ltx-2.5` absent; weights are gitignored and need
`hf auth login` + `hf download` for the gated `Lightricks/LTX-2.5` repo). The
op-level and block-level benchmarks above are the quantitative evidence.

## 8. Git state

- Commit `b316844` "Add xpu-ltx-kernels: SYCL na3d for DiffVAE on Intel XPU
  (B70)" on `main` — changes: new package, `fallback_na/__init__.py`,
  `apply.py`, `AGENTS.md`.
- `origin` = `lipaul/ltx-2-xpu` (existing fork), `sycl-b70` =
  `https://github.com/lipaul/ltx2.5-xpu-sycl-B70` (new target).
- **Push blocked:** the provided PAT authenticates as `li--paul` (id 4507065),
  which has only **pull** access to `lipaul/ltx2.5-xpu-sycl-B70` (`push: false`
  per API). The target repo exists on the `lipaul` account (a different user).
  To push: add `li--paul` as a collaborator with write role on that repo, or
  supply a PAT for the `lipaul` account. The token was used in-memory only,
  never written to disk or the repo.

## 9. Things that would change the approach

- A newer oneAPI + driver than the installed Intel Graphics runtime may fix the
  ESIMD `invalid api option` — revisit then (README notes it).
- BMG FP8 XMX (`blockwise_cpp` scaled GEMM) remains the biggest untapped win if
  someone wants to attempt a full port; needs the fp8 DPAS path and a real
  benchmark vs oneDNN to justify the effort.