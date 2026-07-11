#include "src/model/attention.h"

#include <cmath>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "gtest/gtest.h"
#include "src/model/olmoe_model.h"

namespace nuthatch {
namespace {

OlmoeConfig TinyCfg() {
  OlmoeConfig c;
  c.n_embd = 8;
  c.n_head = 2;
  c.n_head_kv = 2;
  c.n_ff = 4;
  c.n_expert = 4;
  c.n_expert_used = 2;
  c.n_ctx_train = 32;
  c.rope_freq_base = 10000.0f;
  c.rms_eps = 1e-5f;
  return c;
}

// 用可复现的确定性模式填充一个 f32 张量。
void Fill(ggml_tensor* t, float base) {
  float* d = static_cast<float*>(t->data);
  const int64_t n = ggml_nelements(t);
  for (int64_t i = 0; i < n; ++i) {
    d[i] = base + 0.03f * static_cast<float>(i % 13) - 0.15f;
  }
}

// out 布局 [n_embd, T]:元素 (i, token) 在 data[token*n_embd + i]。
float OutAt(ggml_tensor* out, int i, int token, int n_embd) {
  return static_cast<float*>(out->data)[token * n_embd + i];
}

TEST(AttentionTest, ShapeCausalityNoNaN) {
  const OlmoeConfig cfg = TinyCfg();
  const int n_embd = cfg.n_embd, T = 4;

  ggml_init_params ip = {/*mem_size=*/64 * 1024 * 1024, nullptr, false};
  ggml_context* ctx = ggml_init(ip);

  auto v1 = [&](int n, float b) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    Fill(t, b);
    return t;
  };
  auto v2 = [&](int a, int b, float base) {
    ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b);
    Fill(t, base);
    return t;
  };

  AttnWeights w;
  w.attn_norm = v1(n_embd, 1.0f);
  w.q_norm = v1(n_embd, 1.0f);
  w.k_norm = v1(n_embd, 1.0f);
  w.wq = v2(n_embd, n_embd, 0.05f);
  w.wk = v2(n_embd, n_embd, 0.04f);
  w.wv = v2(n_embd, n_embd, 0.06f);
  w.wo = v2(n_embd, n_embd, 0.03f);

  ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T);
  ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);

  ggml_tensor* out = BuildAttention(ctx, cfg, w, x, pos);
  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, out);

  // 输入 + 位置。
  Fill(x, 0.1f);
  int32_t* p = static_cast<int32_t*>(pos->data);
  for (int i = 0; i < T; ++i) p[i] = i;

  ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/1);

  // 形状。
  ASSERT_EQ(out->ne[0], n_embd);
  ASSERT_EQ(out->ne[1], T);

  // 无 NaN/Inf,记下 token 0 的输出。
  std::vector<float> tok0(n_embd);
  for (int i = 0; i < n_embd; ++i) {
    const float o = OutAt(out, i, 0, n_embd);
    ASSERT_TRUE(std::isfinite(o)) << "non-finite at i=" << i;
    tok0[i] = o;
  }

  // 因果:改动【最后一个 token】的输入,重算,token 0 的输出应不变
  //(因果 mask 下 token 0 只 attend 到自己)。
  static_cast<float*>(x->data)[(T - 1) * n_embd + 0] += 7.0f;
  ggml_graph_compute_with_ctx(ctx, gf, 1);
  for (int i = 0; i < n_embd; ++i) {
    EXPECT_FLOAT_EQ(OutAt(out, i, 0, n_embd), tok0[i]) << "causality broken i=" << i;
  }

  // 但改动【token 0 自己】的输入,token 0 输出应改变(健全性)。
  static_cast<float*>(x->data)[(T - 1) * n_embd + 0] -= 7.0f;  // 复原
  static_cast<float*>(x->data)[0 * n_embd + 0] += 7.0f;
  ggml_graph_compute_with_ctx(ctx, gf, 1);
  bool changed = false;
  for (int i = 0; i < n_embd; ++i) {
    if (std::fabs(OutAt(out, i, 0, n_embd) - tok0[i]) > 1e-6f) changed = true;
  }
  EXPECT_TRUE(changed);

  ggml_free(ctx);
}

}  // namespace
}  // namespace nuthatch
