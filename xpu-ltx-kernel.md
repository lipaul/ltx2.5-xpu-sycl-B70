# Porting `ltx-kernels` to Intel XPU (B70): SYCL `na3d` for the DiffVAE decoder

This file documents the end-to-end process of porting a CUDA kernel from
`packages/ltx-kernels` to Intel XPU in a new package, `packages/xpu-ltx-kernels`.
It is a record of decisions, dead ends, and hard-won facts so future porting work
does not repeat the same investigation. The companion living docs are
`packages/xpu-ltx-kernels/README.md` (build/usage) and `AGENTS.md` (toolchain
rules).

Final result: plain-SYCL `na3d` and ESIMD `na3d_esimd` (3D neighborhood
attention) kernels with full correctness parity vs the eager fallback (12 test
cases pass), but **not a performance win for the real workload**. The DiffVAE
decoder's wide kernels `(3,7,7)/(3,5,5)/(11,11,11)` are handled better by
eager's dense batched SDPA than by per-query flash kernels. Measured end-to-end
(512×512/49f, `--offload cpu`): eager decode **7.0s** vs SYCL **17.3s** (~2.5x
slower). The kernels are therefore **opt-in** (`LTX_XPU_NA_KERNELS=1`), not the
XPU default.

**Key outcome of the follow-up investigation:** the earlier claim that "ESIMD
is driver-blocked" was wrong in its conclusion. The blocker was a **oneAPI
version mismatch**: torch 2.12.0+xpu pinned `libsycl.so.8` (2025.3), and the
2025.3 runtime emitted a malformed `-vc-codegen` option token that the driver's
IGC rejected (`invalid api option`). Upgrading to **torch 2.13.0+xpu**
(`libsycl.so.9`, oneAPI 2026.0) and building with the oneAPI 2026.1 compiler
makes ESIMD work end-to-end in the torch process. See §3.6.

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
full additive-mask materialization). That made it an attractive porting target
with a reference implementation in-tree for correctness testing. **Caveat that
only emerged from the end-to-end run:** the decoder's *wide* kernels make eager's
dense batched SDPA faster than a per-query flash kernel on this GPU, so the port
ended up as parity, not a speedup (see §6).

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

### 3.2 ESIMD is rejected by the current driver (resolved in §3.6)

The user explicitly asked for ESIMD/SYCL. We wrote a proper ESIMD kernel
(`esimd::simd` register tiles, `__attribute__((sycl_explicit_simd))`,
`-fsycl-esimd-force-stateless-mem`). Even a trivial `simd<float,16>` copy kernel
fails at JIT:

```
Build program log for 'Intel(R) Graphics [0xe223]': invalid api option: ...
```

AOT (`-fsycl-targets=spir64_gen --device bmg` / `-Xs "-device bmg"`) fails with
`gen compiler command failed` / "No kernel named ... found". **Initial verdict:
the driver rejects ESIMD-generated SPIR-V on this stack**, so plain SYCL was
used. **This conclusion was later corrected** — see §3.6: the real cause was a
oneAPI 2025.3 runtime bug (malformed `-vc-codegen` option), fixed by upgrading
to torch 2.13.0+xpu / oneAPI 2026.1.

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

### 3.6 ESIMD root cause: it was a oneAPI 2025.3 bug, not the driver

Follow-up investigation ("is the problem in oneAPI, torch-xpu, or a version
incompatibility?") answered: **a oneAPI version mismatch**. Determinative
evidence, all reproduced standalone (no torch):

| build toolchain                       | ESIMD result                  |
|---------------------------------------|-------------------------------|
| oneAPI 2025.3 icpx + libsycl.so.8     | `invalid api option` (fails)  |
| oneAPI 2026.1 icpx + libsycl.so.9     | **works** (`h0=1.0` correct)  |
| 2025.3 plain SYCL (non-ESIMD)         | works (control)               |
| 2025.3 ESIMD over OpenCL and Level-Zero | same failure (same IGC)      |

So the driver, the GPU (B70, `05:00.0 Battlemage G31`) and the IGC version all
accept ESIMD — only the 2025.3 toolchain produces a bad program. `IGC_ShaderDumpEnable`
captured the exact IGC build options:

```
2025.3:  \x32\x35 -vc-codegen        <- malformed prefix (\x32\x35 = ASCII "25")
2026.1:   -vc-codegen -disable-finalizer-msg
```

The 2025.3 `libsycl.so.8` serialized a stray option token in front of
`-vc-codegen`, which IGC 2.28.4's strict parser rejects with `invalid api
option`. 2026.1 emits a clean string. **Fix: upgrade torch to 2.13.0+xpu**
(which links `libsycl.so.9`, oneAPI 2026.0 — verified from the wheel's
`libtorch_xpu.so` NEEDED entries and METADATA deps `intel-sycl-rt==2026.0.0`)
and build `xpu-ltx-kernels` with the oneAPI 2026.1 compiler. ESIMD then runs
end-to-end in the torch process (`na3d_esimd` op). AOT (`spir64_gen --device
bmg`) still fails (`gen compiler command failed`, exit 226) on both 2025.3 and
2026.1, but JIT (`spir64`) works — fine for a per-process extension.

Perf caveat: unlocking ESIMD does **not** by itself beat eager. At the decoder's
wide kernels, per-query ESIMD flash (0.1-0.2x) and plain SYCL (0.1-1.0x) are
still slower than eager's dense batched SDPA (a GEMM at ~11-18 TFLOP/s). A real
win needs a batched-GEMM-shaped DPAS kernel; a raw DPAS throughput probe was
launch-bound and inconclusive, and the effort was not pursued. NA stays opt-in.

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

`csrc/na3d_dpas.cpp` — **batched-GEMM DPAS** version (XMX systolic). One
work-item per `(b,t,h,nh,w-slice of 8 queries)`. For each 16-key chunk of the
slice's halo: DPAS QK^T (M=8 × K=16 × N=16, accumulated over HD/16 K-slices),
per-query window mask + online softmax, then DPAS PV (M=8 × 16 keys × HD in
16-slices). **Key discovery — DPAS operand layout (see Phase-0 probes):**
- B must be **VNNI-packed**: for bf16, each `uint32` = `{bf16[2k][n],
  bf16[2k+1][n]}` (two K-rows vertically packed). Without this, general B
  values produce garbage (identity-B tests pass by coincidence; randn inputs
  expose it).
- With VNNI-packed B, the dpas **result is row-major** `S[m][n]` — no
  even/odd interleave (an initial wrong guess that cost debugging time).
- `dpas<8, RepeatCount, float, float, bf16, bf16, bf16, bf16, M*N, BN, AN>(C,
  B, A)` computes `C(M×N) += A(M×16) × B(16×N)`; RepeatCount ≤ 8, N ∈ {8,16}.

Experiments that did **not** win and were reverted:

- **W-blocked kernel** (WB=2/4 W positions per work-item to reuse shared K/V
  columns): register pressure from `WB × HD` fp32 accumulators spilled and
  regressed vs the per-query kernel on small tiles; the per-query kernel won.
- **ESIMD** (see 3.2 — resolved by the torch 2.13 upgrade, but per-query ESIMD
  still loses to eager).
- **na3d_dpas performance**: correct (full parity at `rtol/atol 2e-2`) but
  slower than eager — too many small work-items recomputing the full halo QK^T
  (560 keys) for only 8 queries (147-key window ≈ 4x wasted DPAS), per-query
  softmax on the critical path. Beating eager needs a persistent grid with
  M=64-style tiles and DPAS PV off the softmax path (like `na_attn_dsl`'s
  cooperative tiles).

`csrc/na3d_dpas2.cpp` — **tuned DPAS kernel** (the incremental wins):
- **window-union halo**: the kt/kh/kw halo expansion is unnecessary; the
  kernel only touches the union of the tile's query windows (`w_hi` =
  `window_bounds(w0+MQ-1).end-1`, no `+rw`). This alone cuts the (3,7,7)
  per-tile key volume ~2x.
- **hoisted key decode**: the div/mod chain `kidx/(nkh*nkw)`, `(kidx/nkw)%nkh`,
  `kidx%nkw` was recomputed per (query,key) in both the max-pass and the
  exp-pass. Computing `(kt,kh,kw)` once per 16-key chunk into a register
  array and reusing it across all 8 queries was the single biggest win
  (48ms vs 64ms on (3,7,7) 9x16x16, NH=8).
- **MQ=8 (not 16)**: MQ=16 doubled the live register state (Qreg+Oacc 16×64 +
  per-chunk S 16×16) and spilled; MQ=8 stayed fast. SUB-tiling (2×8 sub-tiles
  sharing one halo load) also regressed — the extra live S/P state cost more
  than the saved loads.
- Result: **0.9x eager** on the real stage-1 kernel (3,7,7) 9x16x16 NH=8
  (48ms vs eager 42ms). Still loses at NH=32 (eager's whole-volume batched
  SDPA: ~2ms) and on small-window kernels (3,5,5)/(11,11,11) — per-query
  DPAS (M<=8) cannot match eager's batching there.

A fixed Cauchy-Schwarz softmax bound (`bound = ||q||·k_bound`) was tried to
make the PV a stateless GEMM (per the CuTe design), but a single global
`k_bound = max||k||` is far looser than the true rowmax for general inputs
and underflows `exp2` to zero → 0/0 NaN. The report's version is only valid
because the DiffVAE's RMS-norm makes `k_bound = sqrt(HD)·max|k_norm_weight|`
tight; plumb that weight through if it is ever needed. Online softmax is the
robust choice.

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
- **Microbenchmark (misleading for the real workload):** at cubic `(3,3,3)`
  kernels the raw op is 2–8x faster than eager (eager's mask materialization
  dominates small kernels). This is why the naive "2–8x" claim was wrong to
  extrapolate.
- **Real decoder kernels (from VAE config `stage_kernels =
  [[3,7,7],[3,7,7],[3,5,5],[3,5,5],[11,11,11]]`):** eager's dense batched SDPA
  (one masked GEMM per window-geometry group) beats the per-query flash kernels.
  Measured on torch 2.13.0+xpu:

  | shape              | kernel      | sycl / eager | esimd / eager |
  |--------------------|-------------|--------------|---------------|
  | (1,9,16,16,32,64)  | (3,7,7)     | 0.5x         | 0.1x          |
  | (1,9,16,16,32,64)  | (3,5,5)     | 1.0x         | 0.2x          |
  | (1,9,32,32,32,64)  | (3,7,7)     | 0.7x         | 0.2x          |
  | (1,9,16,32,32,64)  | (3,7,7)     | 1.0x         | 0.2x          |
  | (1,11,16,16,32,64) | (11,11,11)  | 0.1x         | 0.02x         |

- **End-to-end (512×512/49f, distilled, `--offload cpu`, decode stage timed, on
  torch 2.13.0+xpu):** eager **7.0s** vs SYCL **17.3s** (~2.5x slower). Note:
  the pipeline needs `@torch.inference_mode()` (as the real CLI `main()` has)
  or the text-encoder cache isn't freed and the DiffVAE tiling check fails with
  `usable_bytes=0`.
- **Conclusion:** the SYCL and ESIMD kernels are correctness-parity ports, not
  speedups, for the shipped decoder. A real win needs a batched-GEMM-shaped
  DPAS kernel. Kept **opt-in** via `LTX_XPU_NA_KERNELS=1`; eager remains the XPU
  default.
- `uv run ruff check` clean on all changed files.

## 7. End-to-end run status

Models were present at `/home/acm/work/models/ltx-2.5` (split layout). Full
pipeline A/B ran at 512×512/49f with `--offload cpu` (required for the 22B
model on the 30 GB XPU). Decode-stage timing (wrapping `VideoDecoder.__call__`):
**eager 6.42s vs SYCL 17.85s**. Two run-harness gotchas learned:
- The pipeline must run under `@torch.inference_mode()` (the real CLI `main()`
  is so decorated); a bare `pipeline(...)` call leaves the text-encoder cache
  resident, and the DiffVAE tiling check then reports `usable_bytes=0`
  (`Cannot fit a DiffVAE decode tile`).
- Use `--offload cpu`; without it the 40 GB transformer won't fit the 30 GB GPU.

## 8. Git state

Commits on `main`, all pushed to `sycl-b70`:

- `b316844` "Add xpu-ltx-kernels: SYCL na3d for DiffVAE on Intel XPU (B70)" —
  new package, `fallback_na/__init__.py` + `apply.py` wiring, `AGENTS.md`.
- `057fa14` "Add porting record: xpu-ltx-kernel.md" — this document.
- `9df2044` "Make xpu-ltx-kernels NA opt-in; document honest e2e result" —
  NA is opt-in (`LTX_XPU_NA_KERNELS=1`), e2e harnesses in `bench/`.
- `cbb7c3c` "Upgrade to torch 2.13.0+xpu (libsycl.so.9); unlock ESIMD" —
  torch/torchvision/triton-xpu bump, oneAPI 2026.1 toolchain, `na3d_esimd` op,
  §3.6 (ESIMD root-cause: a oneAPI 2025.3 bug, fixed by the upgrade).

Remotes: `origin` = `lipaul/ltx-2-xpu` (existing fork), `sycl-b70` =
`https://github.com/lipaul/ltx2.5-xpu-sycl-B70` (target). Push is done with a
PAT for the `lipaul` account (the repo owner); the first PAT provided belonged
to `li--paul` which only had pull access — the `lipaul`-account token worked.
Tokens were used in-memory only, never written to disk or the repo (verified no
`ghp_`/`x-access-token` in git config).

## 9. Things that would change the approach

- ESIMD is now unlocked (torch 2.13.0+xpu + oneAPI 2026.1). A **batched-GEMM
  shaped DPAS kernel** for NA is the remaining path to beat eager's dense SDPA
  at the wide decoder kernels; per-query flash (SYCL or ESIMD) cannot.
- BMG FP8 XMX (`blockwise_cpp` scaled GEMM) remains the biggest untapped win if
  someone wants to attempt a full port; needs the fp8 DPAS path and a real
  benchmark vs oneDNN to justify the effort.