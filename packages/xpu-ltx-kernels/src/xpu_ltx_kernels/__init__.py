"""xpu_ltx_kernels: SYCL kernels for ltx-core on Intel XPU (B70).

Delivers a fused 3D neighborhood attention (``na3d``) for the DiffVAE decoder,
replacing the slow eager tiled-SDPA fallback that XPU currently runs (natten is
CUDA-only and the Triton fallback is gated on ``torch.cuda``).

Build the native library (``libxpu_ltx_kernels.so``) with CMake using the
matching oneAPI compiler, then either ``import xpu_ltx_kernels`` (which loads
the library) or call :func:`load()` directly. See the package README.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_LIB_NAME = "libxpu_ltx_kernels.so"
_loaded = False


def _candidates() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [here / _LIB_NAME]


def load() -> bool:
    """Load the native SYCL library. Returns True if it loaded successfully.

    Never raises: failure just means the eager fallback stays in use.
    """
    global _loaded
    if _loaded:
        return True
    for path in _candidates():
        if not path.is_file():
            continue
        try:
            torch.ops.load_library(str(path))
        except (OSError, RuntimeError) as exc:
            logger.warning("xpu_ltx_kernels: failed to load %s: %s", path, exc)
            return False
        _loaded = True
        logger.info("xpu_ltx_kernels: loaded %s", path)
        return True
    logger.debug("xpu_ltx_kernels: %s not found", _LIB_NAME)
    return False


def available() -> bool:
    """True when the native library is loadable and the XPU device is present."""
    if not torch.xpu.is_available():
        return False
    if not load():
        return False
    return hasattr(torch.ops.xpu_ltx_kernels, "na3d")


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
) -> torch.Tensor:
    """3D neighborhood attention (NATTEN ``na3d`` semantics) on XPU.

    ``q/k/v``: ``(B, T, H, W, NH, HD)`` bf16 contiguous on XPU. ``scale`` is
    assumed 1.0 (Q is pre-scaled upstream). Mirrors
    ``ltx_core...fallback_na.eager.na3d`` semantics.
    """
    load()
    return torch.ops.xpu_ltx_kernels.na3d(q, k, v, *kernel_size)


__all__ = ["available", "load", "na3d"]