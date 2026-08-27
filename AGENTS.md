# AGENTS.md

LTX-2: a PyTorch audio-video diffusion model. This working tree is a vendored snapshot of
[`Lightricks/LTX-2`](https://github.com/Lightricks/LTX-2), pushed to a personal fork
(`lipaul/ltx-2-xpu`, `git remote -v`). It is a `uv` workspace monorepo of four packages.

## Per-package instruction files (read these first)

These already encode hard-won, repo-specific guidance. Read the relevant one before editing:

- `packages/ltx-trainer/AGENTS.md` — training package: config conventions, `configs/` doc sync rule,
  frame/resolution constraints (`frames % T == 1`), split-vs-unified checkpoint layouts, testing standards.
- `packages/ltx-pipelines/CLAUDE.md` — inference pipelines: pipeline selection table, sigma schedules,
  LoRA conventions, `utils/blocks.py` memory model, DFR invariants, generated-keyframes rules.
- `packages/ltx-kernels/README.md` and `packages/ltx-core/README.md` — CUDA extension layout and model components.
- `packages/xpu-ltx-kernels/README.md` — fork-only XPU SYCL kernels: build prerequisites (oneAPI 2025.3), the
  `na3d` DiffVAE kernel contract, and the ESIMD limitation.
- For training/fine-tuning/LoRA requests, use the skill at `.claude/skills/train-model/SKILL.md` — it orchestrates
  dataset probing, mode selection, preprocessing, launch, and monitoring (see also `packages/ltx-trainer/AGENTS.md`).

Keep changes to each package consistent with its own instruction file.

## Package layout

- `packages/ltx-core/` — model definitions (transformer, VAEs, vocoder, Gemma text encoder), loading, quantization. Depended on by both other packages.
- `packages/ltx-pipelines/` — ready-made inference pipelines (`ltx_pipelines.*`, run as `python -m ltx_pipelines.<module>`).
- `packages/ltx-trainer/` — LoRA / full fine-tuning (`scripts/train.py`, `configs/*.yaml`).
- `packages/ltx-kernels/` — optional CUDA/C++ extensions (all2all, blockwise FP8/FP6 GEMM, NVFP4, CuTe DSL VAE kernels).
- `packages/xpu-ltx-kernels/` — fork-only SYCL kernels for XPU (DiffVAE `na3d` neighborhood attention). Excluded from the uv workspace like `ltx-kernels`; built with CMake + the oneAPI **2025.3** compiler (see its README).

The fork's inference entrypoint is `./run_pipeline.sh "PROMPT" out.mp4 [args...]` — a thin wrapper over
`uv run python -m ltx_pipelines.distilled` with the split-layout model paths pre-wired (model root default
`/home/acm/paul/models/ltx-2.5`, override via `LTX_MODEL_ROOT`; extra args are forwarded verbatim). It does **not**
add `--offload cpu` for you.

## Toolchain (uv + torch XPU)

This fork runs the inference pipeline on **Intel XPU** (Linux). The environment is resolved entirely via `uv`
against the official PyTorch **XPU** index — no CUDA.

- Use `uv` everywhere: `uv sync`, `uv run`. Do not use bare `pip` or a system python.
- torch/torchaudio are pinned to the **XPU** index in `packages/ltx-core/pyproject.toml`:
  `torch==2.13.0+xpu`, `torchaudio==2.11.0+xpu`, `torchvision==0.28.0+xpu`, plus `triton-xpu==3.7.2` (declared as a
  *direct* dep — uv only applies `[tool.uv.sources]` index mappings to direct deps, so the transitive
  `triton-xpu` would otherwise resolve from PyPI at a conflicting version).
- **`transformers` is capped at `<5.15`** (in `ltx-core` deps). 5.15.0+ breaks the Gemma 4 text encoder build.
- `ltx-core` bounds `requires-python = ">=3.10,<3.14"`: `triton-xpu==3.7.2` has no free-threaded 3.14
  (cp314t) Windows wheel, so an unbounded range makes the universal lock unresolvable.
- **`ltx-trainer` and `ltx-kernels` are excluded from the uv workspace** (`[tool.uv.workspace].exclude`). The
  trainer pins CUDA-only `torchcodec`; the kernels are CUDA extensions. A plain `uv sync` installs only
  `ltx-core` + `ltx-pipelines`, the pipeline stack.
- **The `natten` extra is gone** (the README Quick Start still runs `uv sync --extra natten`) — it was dropped
  when the CUDA-only `natten` package was removed; plain `uv sync` is the correct install command on this fork.
- **SYCL toolchain must match torch-xpu's runtime.** torch **2.13.0+xpu** links `libsycl.so.9` (oneAPI 2026.0).
  Build `xpu-ltx-kernels` with the oneAPI **2026.1** compiler (`intel-oneapi-compiler-dpcpp-cpp-2026.1`, apt);
  CMake auto-selects it. Do NOT downgrade to 2025.3/libsycl.so.8: torch 2.12.0+xpu (libsycl.so.8) is the pairing
  whose ESIMD device code the current driver rejects with `invalid api option` (the 2025.3 runtime emits a
  malformed `-vc-codegen` option token; verified via `IGC_ShaderDumpEnable` — the VC options string is
  `\x32\x35 -vc-codegen` vs 2026.1's clean `-vc-codegen -disable-finalizer-msg`). With 2026.1, **ESIMD works**
  end-to-end in the torch process (`na3d_esimd` op).

## Code quality gates

Ruff is configured at the root (`pyproject.toml` `[tool.ruff]`): line length **120**, strict ruleset
(ANN, B, PL, SIM, ...), `known-first-party = ["ltx_core", "ltx_pipelines", "ltx_trainer"]`. Run:

```bash
uv run ruff check .
uv run ruff format .
```

There is **no `.pre-commit-config.yaml`** in this tree — `pre-commit run` is not a working gate; ruff is the gate.

## Runtime gotchas

- Model checkpoints (`.safetensors`) and media are gitignored — never commit them. Downloads and pipeline
  runs need weights fetched via `hf download`/`hf auth login` (gated repo, Read token).
- Checkpoint paths are local only (URLs unsupported). Two layouts exist — **unified** (one `.safetensors` +
  Gemma dir) and **split** (one file per component); they are not interchangeable and the code detects the
  layout from checkpoint metadata, not a flag.
- Audio-video work needs an Intel GPU with XPU support; the pipeline selects the device in
  `packages/ltx-core/src/ltx_core/devices.py` (XPU is probed via `torch.xpu.is_available()`).
- LTX 2.5 requires the LTX-fine-tuned Gemma 4 text encoder; Google's vanilla Gemma 4 is not a substitute.
- **DiffVAE neighborhood attention runs the eager tiled-SDPA fallback by default on XPU** (natten is CUDA-only,
  `triton_na_available()` is gated on `torch.cuda`). `packages/xpu-ltx-kernels` provides a SYCL `na3d` kernel
  (plus an ESIMD `na3d_esimd` that works since the torch 2.13 / oneAPI 2026.1 upgrade), but both are **opt-in**
  (`LTX_XPU_NA_KERNELS=1`): at the decoder's real wide kernels `(3,7,7)/(3,5,5)/(11,11,11)` eager's dense batched
  SDPA is ~2.5x faster end-to-end than the per-query kernels (measured decode 7.0s eager vs 17.3s SYCL at
  512x512/49f). A real win would need a batched-GEMM-shaped (DPAS) kernel, not per-query flash. Keep eager as the
  default.
- **XPU allocator cache is never freed by the upstream memory helpers.** `cleanup_accelerator_memory` /
  `empty_device_cache` / `synchronize_device` in `devices.py` were CUDA/MPS-only, so the ~29 GB XPU
  caching-allocator cache from the text encoder stayed reserved and the DiffVAE tiling check reported
  `usable_bytes=0` (`Cannot fit a DiffVAE decode tile`). `xpu_activation_budget_bytes()` was added (mirrors
  `cuda_activation_budget_bytes`) and the sync/empty helpers now handle XPU via `torch.xpu.empty_cache()`.
- **XPU SDPA falls to the quadratic math backend by default.** torch's XPU runtime-disables the memory-efficient
  kernel, and the old XPU branch in `_sdpa_full_priority()` (`attention.py`) put `EFFICIENT_ATTENTION` first,
  so video attention materialized the full QK^T score matrix and OOM'd (~72 GB at 1536x1024/121f).
  `FLASH_ATTENTION` IS supported on XPU for bf16 — the fix prioritizes `FLASH_ATTENTION>EFFICIENT_ATTENTION>MATH`
  on XPU, cutting attention memory to ~0.17 GB. The warning "Memory Efficient ... falling back to math" will
  still appear for float32 autocast-disabled paths; bf16 flash is what matters.
- **The 22B distilled transformer does not fit on a 32 GB XPU at once (44 GB bf16 weights).** Run pipelines
  with `--offload cpu` (needs ~36 GB RAM for pinned weights; disk is the lowest-memory option) so weights are
  streamed layer-by-layer. Activations then fit under flash attention.
- `torchvision` is required (transitively by transformers' Gemma 4 image processor) but was missing from the
  lockfile; pinned on the XPU index as `torchvision==0.27.0+xpu` (0.27.1 requires torch 2.12.1, conflicting
  with the `torch==2.12.0+xpu` pin).
- Sourcing oneAPI (`/opt/intel/oneapi/setvars.sh`) is required before device tools (`xpu-smi`) see the GPU;
  `sycl-ls` is the reliable probe for the Level-Zero devices.

## Tests

No `test_*.py` files are present in this vendored tree, so `pytest` is not runnable here. If you add code,
follow the testing standards in `packages/ltx-trainer/AGENTS.md` (flat `test_*` functions, test public
interfaces only, behavioral tests over config-only tests).
