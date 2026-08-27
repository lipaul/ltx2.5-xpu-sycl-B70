"""E2E decode-stage timing for the distilled pipeline, NA backend toggleable.

Usage (mirrors the real CLI args; requires LTX_MODEL_ROOT):

    # SYCL kernel (opt-in)
    LTX_MODEL_ROOT=... USE_SYCL=1 \
      python bench/e2e_decode_timing.py --transformer-path ... --offload cpu ...

    # eager tiled-SDPA fallback (default)
    LTX_MODEL_ROOT=... USE_SYCL=0 \
      python bench/e2e_decode_timing.py --transformer-path ... --offload cpu ...

Prints total wall time, the VAE decode-stage time (where NA lives), and a
tiling-memory probe. NOTE: the pipeline must run under torch.inference_mode()
(as the real CLI main() does) or the text-encoder cache is not freed and the
DiffVAE tiling check fails with ``usable_bytes=0``.

Measured at 512x512/49f with --offload cpu: eager decode 6.4s vs SYCL 17.9s.
"""

import logging
import os
import sys
import time

import torch

sys.path.insert(0, "/home/acm/paul_arc/test/ltx-2-xpu/packages/ltx-core/src")
sys.path.insert(0, "/home/acm/paul_arc/test/ltx-2-xpu/packages/ltx-pipelines/src")

logging.basicConfig(level=logging.INFO)

from ltx_core.devices import xpu_activation_budget_bytes
from ltx_pipelines.utils import helpers


def main() -> None:
    use_sycl = os.environ.get("USE_SYCL", "1") == "1"
    import ltx_pipelines.distilled as mod

    # Spy on the tiling budget computation.
    _orig = helpers.tiling_config_for_vae

    def spy_tiling(*a, **k):
        dev = k.get("device")
        print(
            f"[probe] tiling: dev={dev} free={xpu_activation_budget_bytes(dev)/2**30:.2f}GB "
            f"reserved={torch.xpu.memory_reserved(0)/2**30:.2f}GB "
            f"alloc={torch.xpu.memory_allocated(0)/2**30:.2f}GB",
            flush=True,
        )
        return _orig(*a, **k)

    helpers.tiling_config_for_vae = spy_tiling

    params = mod.resolve_cli_params(distilled=True)
    parser = mod.add_generated_keyframes_arg(
        mod.default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True)
    )
    args = parser.parse_args(sys.argv[1:])

    pipeline = mod.DistilledPipeline(
        model_paths=args.model_paths,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        diffvae_optimization=args.diffvae_optimization,
    )

    import xpu_ltx_kernels as _xk

    if use_sycl and not _xk.available():
        raise SystemExit("[e2e] USE_SYCL=1 but xpu_ltx_kernels not available")

    from ltx_pipelines.utils.blocks import VideoDecoder

    _orig_call = VideoDecoder.__call__
    decode_time = {"t": 0.0}

    def timed_call(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            yield from _orig_call(self, *args, **kwargs)
        finally:
            decode_time["t"] += time.perf_counter() - t0

    VideoDecoder.__call__ = timed_call

    t0 = time.perf_counter()
    hdr = mod.resolve_hdr_color_space(images=args.images, hdr=args.hdr)
    vae_dtype = mod.vae_dtype_for_hdr(hdr, torch.bfloat16)
    with torch.inference_mode():
        video, audio, num_frames, tiling_config = pipeline(
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=args.images,
            vae_dtype=vae_dtype,
            color_space=hdr,
            enhance_prompt=args.enhance_prompt,
            enhance_static_cache=args.enhance_static_cache,
            tiling_config=mod.AUTO_TILING,
            generated_keyframes=args.num_generated_keyframes,
        )
    torch.xpu.synchronize()
    total = time.perf_counter() - t0

    out = os.environ.get("OUT", args.output_path)
    mod.encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=out,
        video_chunks_number=mod.get_video_chunks_number(num_frames, tiling_config),
        color_space=hdr,
    )
    print(f"[e2e] backend={'SYCL' if use_sycl else 'eager'} total={total:.2f}s decode={decode_time['t']:.2f}s", flush=True)


if __name__ == "__main__":
    main()