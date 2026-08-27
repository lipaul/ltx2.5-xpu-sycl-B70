// NA3D: fused 3D neighborhood attention for the DiffVAE decoder. SYCL port of
// the ltx_kernels na_attn_dsl role (Blackwell-only upstream) for Intel XPU.
// Mirrors the NATTEN na3d semantics implemented by ltx_core's eager fallback:
//
//   q/k/v: (B, T, H, W, NH, HD) bf16, contiguous
//   per-axis non-causal window: start = min(max(i - k/2, 0), L - k), end = start + k
//   out:   (B, T, H, W, NH, HD) bf16
//
// scale is 1.0 at the call site (Q is pre-scaled by head_dim**-0.5 upstream).
// Accumulation is fp32; output rounded back to bf16.
//
// Strategy (Phase 1, plain SYCL): one work-item per (b, t, h, nh) "row" walks
// the W axis; per query it streams the (kt*kh*kw) key/value vectors from global
// memory with sycl::vec<bf16, 8> loads and a flash-style online softmax so the
// window is never materialized. Grid: (W, B*T*H*NH).

#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <ATen/ATen.h>
#include <sycl/sycl.hpp>
#include <cstring>

namespace xpu_ltx_kernels::na3d {

// bf16 handled as raw uint16_t in device code: the sycl::ext::oneapi::bfloat16
// conversion path crashes on this XPU stack, so we bit-twiddle instead. bf16 is
// the top 16 bits of f32 (no NaN round-trip concern for attention values).
using bf16 = uint16_t;

__inline__ float bf16_to_f32(bf16 h) {
  const uint32_t bits = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

__inline__ bf16 f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  const uint32_t r = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
  return (bf16)r;
}

struct Window {
  int start;
  int end;
};

inline Window window_bounds(int i, int length, int kernel) {
  // Non-causal: centered, clamped at borders. Mirrors eager._window_bounds.
  const int k = kernel < length ? kernel : length;
  const int lo = length - k;
  const int half = k / 2;
  const int start = i - half < 0 ? 0 : (i - half > lo ? lo : i - half);
  return {start, start + k};
}

// Offset of a spatial position in the (B,T,H,W,NH,HD) channel-last layout.
inline int64_t offset_of(int b, int t, int h, int w, int nh, int T, int H, int W,
                         int NH, int HD) {
  return ((((int64_t)b * T + t) * H + h) * W + w) * (int64_t)NH * HD + nh * HD;
}

// dot(q, k) over HDP dims with vectorized loads, fp32 accumulate. The unrolled
// loop lets the compiler emit vector loads; avoiding sycl::vec::load sidesteps
// bf16 element-API friction across SYCL versions.
template <int HDP>
inline float dot(const float qv[HDP / 8][8], const bf16* __restrict k) {
  float acc = 0.0f;
#pragma unroll
  for (int d = 0; d < HDP; d += 8) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc += qv[d / 8][j] * bf16_to_f32(k[d + j]);
    }
  }
  return acc;
}

// acc += v * p  (flash-weighted value accumulation).
template <int HDP>
inline void fma_acc(float acc[HDP / 8][8], const bf16* __restrict v, float p) {
#pragma unroll
  for (int d = 0; d < HDP; d += 8) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc[d / 8][j] += bf16_to_f32(v[d + j]) * p;
    }
  }
}

// Load query into a register tile.
template <int HDP>
inline void load_query(float qv[HDP / 8][8], const bf16* __restrict q) {
#pragma unroll
  for (int d = 0; d < HDP; d += 8) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      qv[d / 8][j] = bf16_to_f32(q[d + j]);
    }
  }
}

template <int HDP>
inline void scale_acc(float acc[HDP / 8][8], float s) {
#pragma unroll
  for (int d = 0; d < HDP / 8; ++d) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc[d][j] *= s;
    }
  }
}

template <int HDP>
inline void store_acc(bf16* __restrict out, const float acc[HDP / 8][8]) {
#pragma unroll
  for (int d = 0; d < HDP; d += 8) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      out[d + j] = f32_to_bf16(acc[d / 8][j]);
    }
  }
}

// One work-item handles one (b,t,h,nh) row across all W positions for a single
// (w) column assigned to this thread.
template <int HDP>
void na3d_row_kernel(
    const bf16* __restrict q,
    const bf16* __restrict k,
    const bf16* __restrict v,
    bf16* __restrict out,
    int T,
    int H,
    int W,
    int NH,
    int kt,
    int kh,
    int kw,
    int total_rows,
    sycl::id<2> gid) {
  const int row = gid[1];
  const int w = gid[0];
  if (row >= total_rows || w >= W) return;

  int r = row;
  const int nh = r % NH;
  r /= NH;
  const int h = r % H;
  r /= H;
  const int t = r % T;
  r /= T;
  const int b = r;

  const auto wt = window_bounds(t, T, kt);
  const auto wh = window_bounds(h, H, kh);
  const auto ww = window_bounds(w, W, kw);

  const int64_t qoff = offset_of(b, t, h, w, nh, T, H, W, NH, HDP);
  const bf16* qptr = q + qoff;
  bf16* optr = out + qoff;

  float qv[HDP / 8][8];
  load_query<HDP>(qv, qptr);

  // Flash-style online softmax over the kt*kh*kw window.
  float m = -INFINITY;
  float l = 0.0f;
  float acc[HDP / 8][8] = {};

  for (int tt = wt.start; tt < wt.end; ++tt) {
    for (int hh = wh.start; hh < wh.end; ++hh) {
      for (int wl = ww.start; wl < ww.end; ++wl) {
        const int64_t koff = offset_of(b, tt, hh, wl, nh, T, H, W, NH, HDP);
        const float s = dot<HDP>(qv, k + koff);
        const float m_new = sycl::fmax(m, s);
        const float alpha = sycl::exp(m - m_new);
        const float p = sycl::exp(s - m_new);
        l = l * alpha + p;
        scale_acc<HDP>(acc, alpha);
        fma_acc<HDP>(acc, v + koff, p);
        m = m_new;
      }
    }
  }

  scale_acc<HDP>(acc, 1.0f / l);
  store_acc<HDP>(optr, acc);
}

template <int HDP>
void launch_na3d(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    int kt,
    int kh,
    int kw,
    sycl::queue& queue) {
  const int B = q.size(0);
  const int T = q.size(1);
  const int H = q.size(2);
  const int W = q.size(3);
  const int NH = q.size(4);

  const int total_rows = B * T * H * NH;
  auto* qp = reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>());
  auto* kp = reinterpret_cast<const bf16*>(k.data_ptr<at::BFloat16>());
  auto* vp = reinterpret_cast<const bf16*>(v.data_ptr<at::BFloat16>());
  auto* op = reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>());

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<2>(W, total_rows), [=](sycl::id<2> gid) {
      na3d_row_kernel<HDP>(qp, kp, vp, op, T, H, W, NH, kt, kh, kw,
                           total_rows, gid);
    });
  });
}

at::Tensor na3d(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    int64_t kt,
    int64_t kh,
    int64_t kw) {
  TORCH_CHECK(q.device().is_xpu(), "xpu_ltx_kernels::na3d requires XPU tensors");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "xpu_ltx_kernels::na3d expects bf16");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shapes must match");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "xpu_ltx_kernels::na3d expects contiguous inputs");
  TORCH_CHECK(q.dim() == 6, "expected (B,T,H,W,NH,HD)");
  TORCH_CHECK(kt > 0 && kh > 0 && kw > 0, "kernel sizes must be positive");

  const int T = q.size(1), H = q.size(2), W = q.size(3), HD = q.size(5);
  TORCH_CHECK(T >= kt && H >= kh && W >= kw, "spatial dims must be >= kernel sizes");

  auto out = at::empty_like(q);
  sycl::queue& queue = c10::xpu::getCurrentXPUStream();

  if (HD == 64) {
    launch_na3d<64>(q, k, v, out, (int)kt, (int)kh, (int)kw, queue);
  } else if (HD == 32) {
    launch_na3d<32>(q, k, v, out, (int)kt, (int)kh, (int)kw, queue);
  } else {
    TORCH_CHECK(false, "xpu_ltx_kernels::na3d supports HD 32 or 64, got ", HD);
  }
  return out;
}

}  // namespace xpu_ltx_kernels::na3d

TORCH_LIBRARY(xpu_ltx_kernels, m) {
  m.def("na3d(Tensor q, Tensor k, Tensor v, int kt, int kh, int kw) -> Tensor",
        &xpu_ltx_kernels::na3d::na3d);
}