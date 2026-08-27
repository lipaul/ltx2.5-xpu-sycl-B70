// na3d_dpas: batched-GEMM-shaped 3D neighborhood attention via ESIMD DPAS.
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

namespace xpu_ltx_kernels::na3d_dpas {
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

inline float bf16_to_f32(bf16 h) { return (float)h; }
inline bf16 f32_to_bf16(float f) { return bf16(f); }
inline uint16_t bf16_bits(bf16 h) {
  const float f = (float)h;
  return (uint16_t)(sycl::bit_cast<uint32_t>(f) >> 16);
}

struct Window { int start; int end; };
inline Window window_bounds(int i, int length, int kernel) {
  const int k = kernel < length ? kernel : length;
  const int lo = length - k;
  const int half = k / 2;
  const int start = i - half < 0 ? 0 : (i - half > lo ? lo : i - half);
  return {start, start + k};
}
inline int64_t offset_of(int b, int t, int h, int w, int nh, int T, int H, int W, int NH, int HD) {
  return ((((int64_t)b * T + t) * H + h) * W + w) * (int64_t)NH * HD + nh * HD;
}

template <int HD, int MQ, int NK>
void na3d_dpas_slice(
    const bf16* __restrict q, const bf16* __restrict k, const bf16* __restrict v,
    bf16* __restrict out, int T, int H, int W, int NH, int kt, int kh, int kw,
    int b, int t, int h, int w0, int nh) {
  using namespace esimd;
  const int rt = kt / 2, rh = kh / 2, rw = kw / 2;
  const auto wt = window_bounds(t, T, kt);
  const auto wh = window_bounds(h, H, kh);
  const int w_lo = window_bounds(w0, W, kw).start - rw < 0 ? 0 : (window_bounds(w0, W, kw).start - rw);
  const int w_hi_raw = window_bounds(w0 + MQ - 1, W, kw).end + rw - 1;
  const int w_hi = w_hi_raw >= W ? W - 1 : w_hi_raw;

  simd<float, MQ * HD> Qreg = 0.0f;
#pragma unroll
  for (int m = 0; m < MQ; ++m) {
    const int gw = w0 + m;
    if (gw >= W) continue;
    const int64_t qoff = offset_of(b, t, h, gw, nh, T, H, W, NH, HD);
    for (int hd = 0; hd < HD; ++hd) Qreg[m * HD + hd] = bf16_to_f32(q[qoff + hd]);
  }

  simd<float, MQ> m_ = -INFINITY;
  simd<float, MQ> l = 0.0f;
  simd<float, MQ * HD> Oacc = 0.0f;
  float mv[MQ];
  float Pw[MQ * NK];

  const int nkt = wt.end - wt.start;
  const int nkh = wh.end - wh.start;
  const int nkw = w_hi - w_lo + 1;
  const int nbox = nkt * nkh * nkw;

  for (int key0 = 0; key0 < nbox; key0 += NK) {
    simd<bf16, NK * HD> Kreg = 0;
    simd<bf16, NK * HD> Vreg = 0;
    for (int j = 0; j < NK; ++j) {
      const int kidx = key0 + j;
      if (kidx >= nbox) break;
      const int kt_ = wt.start + kidx / (nkh * nkw);
      const int kh_ = wh.start + (kidx / nkw) % nkh;
      const int kw_ = w_lo + kidx % nkw;
      const int ckt = kt_ < 0 ? 0 : (kt_ >= T ? T - 1 : kt_);
      const int ckh = kh_ < 0 ? 0 : (kh_ >= H ? H - 1 : kh_);
      const int ckw = kw_ < 0 ? 0 : (kw_ >= W ? W - 1 : kw_);
      const int64_t koff = offset_of(b, ckt, ckh, ckw, nh, T, H, W, NH, HD);
      for (int hd = 0; hd < HD; ++hd) { Kreg[j * HD + hd] = k[koff + hd]; Vreg[j * HD + hd] = v[koff + hd]; }
    }

    simd<float, MQ * NK> S = 0.0f;
    for (int ks = 0; ks < HD / 16; ++ks) {
      simd<bf16, MQ * 16> Asub;
#pragma unroll
      for (int m = 0; m < MQ; ++m)
#pragma unroll
        for (int j = 0; j < 16; ++j) Asub[m * 16 + j] = f32_to_bf16(Qreg[m * HD + ks * 16 + j]);
      simd<uint32_t, 8 * NK> Bvnni;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int n = 0; n < NK; ++n) {
          const uint16_t lo = bf16_bits(Kreg[n * HD + ks * 16 + 2 * j]);
          const uint16_t hi = bf16_bits(Kreg[n * HD + ks * 16 + 2 * j + 1]);
          Bvnni[j * NK + n] = (uint32_t)lo | ((uint32_t)hi << 16);
        }
      simd<bf16, 16 * NK> Bsub = Bvnni.template bit_cast_view<bf16>();
      S = xmx::dpas<8, MQ, float, float, bf16, bf16, xmx::dpas_argument_type::bf16,
                    xmx::dpas_argument_type::bf16, MQ * NK, 16 * NK, MQ * 16>(S, Bsub, Asub);
    }

#pragma unroll
    for (int m = 0; m < MQ; ++m) {
      const int gw = w0 + m;
      if (gw >= W) continue;
      const auto wqt = window_bounds(t, T, kt);
      const auto wqh = window_bounds(h, H, kh);
      const auto wqw = window_bounds(gw, W, kw);
      float m_new = -INFINITY;
#pragma unroll
      for (int j = 0; j < NK; ++j) {
        const int kidx = key0 + j;
        if (kidx >= nbox) { Pw[m * NK + j] = -INFINITY; continue; }
        const int kt_ = wt.start + kidx / (nkh * nkw);
        const int kh_ = wh.start + (kidx / nkw) % nkh;
        const int kw_ = w_lo + kidx % nkw;
        const bool in_win = (kt_ >= wqt.start && kt_ < wqt.end &&
                             kh_ >= wqh.start && kh_ < wqh.end &&
                             kw_ >= wqw.start && kw_ < wqw.end);
        const float s = in_win ? S[m * NK + j] : -INFINITY;
        m_new = m_new > s ? m_new : s;
        Pw[m * NK + j] = s;
      }
      mv[m] = m_new;
    }
    bool any = false;
#pragma unroll
    for (int m = 0; m < MQ; ++m) if (mv[m] != -INFINITY) { any = true; break; }
    if (!any) continue;
    simd<float, MQ * NK> Pv = 0.0f;
#pragma unroll
    for (int m = 0; m < MQ; ++m) {
      if (mv[m] == -INFINITY) continue;
      float l_new = 0.0f;
      float Prow[NK];
#pragma unroll
      for (int j = 0; j < NK; ++j) {
        if (Pw[m * NK + j] == -INFINITY) { Prow[j] = 0.0f; continue; }
        Prow[j] = exp(Pw[m * NK + j] - mv[m]);
        l_new += Prow[j];
      }
      const float alpha = exp(m_[m] - mv[m]);
      l[m] = l[m] * alpha + l_new;
      m_[m] = mv[m];
#pragma unroll
      for (int hd = 0; hd < HD; ++hd) Oacc[m * HD + hd] *= alpha;
#pragma unroll
      for (int j = 0; j < NK; ++j) Pv[m * NK + j] = Prow[j];
    }
    // Batched DPAS PV: Oacc += P x V
    simd<bf16, MQ * NK> Preg;
#pragma unroll
    for (int m = 0; m < MQ; ++m)
#pragma unroll
      for (int j = 0; j < NK; ++j) Preg[m * NK + j] = f32_to_bf16(Pv[m * NK + j]);
    for (int ks = 0; ks < HD / 16; ++ks) {
      simd<bf16, MQ * 16> Asub;
#pragma unroll
      for (int m = 0; m < MQ; ++m)
#pragma unroll
        for (int j = 0; j < 16; ++j) Asub[m * 16 + j] = Preg[m * NK + j];
      simd<uint32_t, 8 * 16> Bvnni;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int hd = 0; hd < 16; ++hd) {
          const uint16_t lo = bf16_bits(Vreg[(2 * j) * HD + ks * 16 + hd]);
          const uint16_t hi = bf16_bits(Vreg[(2 * j + 1) * HD + ks * 16 + hd]);
          Bvnni[j * 16 + hd] = (uint32_t)lo | ((uint32_t)hi << 16);
        }
      simd<bf16, 16 * 16> Bsub = Bvnni.template bit_cast_view<bf16>();
      simd<float, MQ * 16> Os;
#pragma unroll
      for (int m = 0; m < MQ; ++m)
#pragma unroll
        for (int hd = 0; hd < 16; ++hd) Os[m * 16 + hd] = Oacc[m * HD + ks * 16 + hd];
      Os = xmx::dpas<8, MQ, float, float, bf16, bf16, xmx::dpas_argument_type::bf16,
                     xmx::dpas_argument_type::bf16, MQ * 16, 16 * 16, MQ * 16>(Os, Bsub, Asub);
#pragma unroll
      for (int m = 0; m < MQ; ++m)
#pragma unroll
        for (int hd = 0; hd < 16; ++hd) Oacc[m * HD + ks * 16 + hd] = Os[m * 16 + hd];
    }
  }

#pragma unroll
  for (int m = 0; m < MQ; ++m) {
    const int gw = w0 + m;
    if (gw >= W) continue;
    const int64_t ooff = offset_of(b, t, h, gw, nh, T, H, W, NH, HD);
    const float inv = 1.0f / l[m];
#pragma unroll
    for (int hd = 0; hd < HD; ++hd) out[ooff + hd] = f32_to_bf16(Oacc[m * HD + hd] * inv);
  }
}

at::Tensor na3d_dpas(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, int64_t kt, int64_t kh, int64_t kw) {
  TORCH_CHECK(q.device().is_xpu(), "xpu_ltx_kernels::na3d_dpas requires XPU tensors");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "expects bf16");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shapes must match");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "expects contiguous");
  TORCH_CHECK(q.dim() == 6, "expected (B,T,H,W,NH,HD)");
  TORCH_CHECK(kt > 0 && kh > 0 && kw > 0, "kernel sizes must be positive");
  TORCH_CHECK(q.size(5) == 64, "xpu_ltx_kernels::na3d_dpas supports HD=64");
  const int B = q.size(0), T = q.size(1), H = q.size(2), W = q.size(3), NH = q.size(4);
  TORCH_CHECK(T >= kt && H >= kh && W >= kw, "spatial dims must be >= kernel sizes");
  auto out = at::empty_like(q);
  sycl::queue& queue = c10::xpu::getCurrentXPUStream();
  auto* qp = reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>());
  auto* kp = reinterpret_cast<const bf16*>(k.data_ptr<at::BFloat16>());
  auto* vp = reinterpret_cast<const bf16*>(v.data_ptr<at::BFloat16>());
  auto* op = reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>());
  constexpr int MQ = 8, NK = 16;
  const int nw = (W + MQ - 1) / MQ;
  const int total = B * T * H * NH * nw;
  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
      int idx = gid[0];
      const int wq = idx % nw; idx /= nw;
      const int nh = idx % NH; idx /= NH;
      const int h = idx % H; idx /= H;
      const int t = idx % T; idx /= T;
      const int b = idx;
      na3d_dpas_slice<64, MQ, NK>(qp, kp, vp, op, T, H, W, NH, (int)kt, (int)kh, (int)kw, b, t, h, wq * MQ, nh);
    });
  });
  return out;
}
}  // namespace xpu_ltx_kernels::na3d_dpas
