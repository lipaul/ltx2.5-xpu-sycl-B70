// ESIMD NA3D (experimental): 3D neighborhood attention with Intel Explicit SIMD.
//
// Same semantics as the plain-SYCL na3d (na3d.cpp). Uses esimd::simd register
// tiles for the query/accumulator and per-column K/V loads. Historically this
// was driver-blocked ("invalid api option") when built with oneAPI 2025.3
// (libsycl.so.8) -- the 2025.3 runtime emitted a malformed "-vc-codegen" option
// token. With oneAPI 2026.x (libsycl.so.9, torch 2.13.0+xpu) the ESIMD device
// code is accepted, so this path is viable again. Keep it behind an explicit
// entry point (na3d_esimd op); production dispatch stays with na3d until DPAS
// wins on the decoder's wide kernels.

#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <ATen/ATen.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

namespace xpu_ltx_kernels::na3d_esimd {

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
  const int k = kernel < length ? kernel : length;
  const int lo = length - k;
  const int half = k / 2;
  const int start = i - half < 0 ? 0 : (i - half > lo ? lo : i - half);
  return {start, start + k};
}

inline int64_t offset_of(int b, int t, int h, int w, int nh, int T, int H, int W,
                         int NH, int HD) {
  return ((((int64_t)b * T + t) * H + h) * W + w) * (int64_t)NH * HD + nh * HD;
}

// ESIMD kernel: one work-item per (b,t,h,nh) row and a small block of W
// positions. Q lives in esimd::simd register tiles; each K/V column of the
// shared (t,h) window is loaded once (block_load) and reused across the block's
// queries. fp32 accumulators in simd<float, W_CHUNK> per head dim.
template <int HDP>
ESIMD_INLINE void na3d_esimd_row(
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
    int row,
    int w0) {
  using namespace sycl::ext::intel::esimd;

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
  const auto ww = window_bounds(w0, W, kw);

  const int64_t qoff = offset_of(b, t, h, w0, nh, T, H, W, NH, HDP);
  simd<float, HDP> qv;
#pragma unroll
  for (int d = 0; d < HDP; ++d) {
    qv[d] = bf16_to_f32(q[qoff + d]);
  }

  float m = -INFINITY;
  float l = 0.0f;
  simd<float, HDP> acc = 0.0f;

  for (int tt = wt.start; tt < wt.end; ++tt) {
    for (int hh = wh.start; hh < wh.end; ++hh) {
      for (int wcol = ww.start; wcol < ww.end; ++wcol) {
        const int64_t koff = offset_of(b, tt, hh, wcol, nh, T, H, W, NH, HDP);
        simd<float, HDP> kv;
#pragma unroll
        for (int d = 0; d < HDP; ++d) {
          kv[d] = bf16_to_f32(k[koff + d]);
        }
        const simd<float, HDP> prod = qv * kv;
        float s = 0.0f;
#pragma unroll
        for (int d = 0; d < HDP; ++d) {
          s += prod[d];
        }
        const float m_new = max(m, s);
        const float alpha = exp(m - m_new);
        const float p = exp(s - m_new);
        l = l * alpha + p;
        acc *= alpha;
#pragma unroll
        for (int d = 0; d < HDP; ++d) {
          acc[d] += bf16_to_f32(v[koff + d]) * p;
        }
        m = m_new;
      }
    }
  }

  acc /= l;
#pragma unroll
  for (int d = 0; d < HDP; ++d) {
    out[qoff + d] = f32_to_bf16(acc[d]);
  }
}

template <int HDP>
void launch_na3d_esimd(
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
    h.parallel_for(sycl::range<2>(W, total_rows), [=](sycl::id<2> gid)
                       __attribute__((sycl_explicit_simd)) {
      na3d_esimd_row<HDP>(qp, kp, vp, op, T, H, W, NH, kt, kh, kw, gid[1],
                          gid[0]);
    });
  });
}

at::Tensor na3d_esimd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    int64_t kt,
    int64_t kh,
    int64_t kw) {
  TORCH_CHECK(q.device().is_xpu(), "xpu_ltx_kernels::na3d_esimd requires XPU tensors");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "xpu_ltx_kernels::na3d_esimd expects bf16");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shapes must match");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "xpu_ltx_kernels::na3d_esimd expects contiguous inputs");
  TORCH_CHECK(q.dim() == 6, "expected (B,T,H,W,NH,HD)");
  TORCH_CHECK(kt > 0 && kh > 0 && kw > 0, "kernel sizes must be positive");

  const int T = q.size(1), H = q.size(2), W = q.size(3), HD = q.size(5);
  TORCH_CHECK(T >= kt && H >= kh && W >= kw, "spatial dims must be >= kernel sizes");

  auto out = at::empty_like(q);
  sycl::queue& queue = c10::xpu::getCurrentXPUStream();

  if (HD == 64) {
    launch_na3d_esimd<64>(q, k, v, out, (int)kt, (int)kh, (int)kw, queue);
  } else if (HD == 32) {
    launch_na3d_esimd<32>(q, k, v, out, (int)kt, (int)kh, (int)kw, queue);
  } else {
    TORCH_CHECK(false, "xpu_ltx_kernels::na3d_esimd supports HD 32 or 64, got ", HD);
  }
  return out;
}

}  // namespace xpu_ltx_kernels::na3d_esimd