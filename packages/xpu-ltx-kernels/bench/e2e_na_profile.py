import logging
import os
import sys
import time

import torch

sys.path.insert(0, "/home/acm/paul_arc/test/ltx-2-xpu/packages/ltx-core/src")
sys.path.insert(0, "/home/acm/paul_arc/test/ltx-2-xpu/packages/ltx-pipelines/src")

logging.basicConfig(level=logging.INFO)



def main() -> None:
    import ltx_pipelines.distilled as mod

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

    # Instrument every NA call: shape + cumulative time per backend.
    from ltx_core.model.video_vae.transformer.fallback_na import (
        EagerSdpaAttention,
        XpuNaAttention,
    )

    stats = {"eager": [0, 0.0], "xpu": [0, 0.0]}  # count, total seconds

    def make_probe(backend_name, inner):
        def probe(self, attn, q, k, v):
            t0 = time.perf_counter()
            out = inner(self, attn, q, k, v)
            torch.xpu.synchronize()
            dt = time.perf_counter() - t0
            stats[backend_name][0] += 1
            stats[backend_name][1] += dt
            if stats[backend_name][0] <= 3:
                print(
                    f"[shape] {backend_name} q={tuple(q.shape)} k={tuple(k.shape)} "
                    f"ks={attn.kernel_size} dt={dt*1e3:.1f}ms",
                    flush=True,
                )
            return out

        return probe

    EagerSdpaAttention.__call__ = make_probe("eager", EagerSdpaAttention.__call__)
    XpuNaAttention.__call__ = make_probe("xpu", XpuNaAttention.__call__)

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
    print(f"[e2e] total={total:.2f}s decode={decode_time['t']:.2f}s", flush=True)
    for name, (n, t) in stats.items():
        print(f"[stats] {name}: calls={n} total={t:.2f}s avg={t/max(1,n)*1e3:.3f}ms", flush=True)


if __name__ == "__main__":
    main()