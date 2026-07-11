#include "src/model/moe.h"

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
  c.rms_eps = 1e-5f;
  return c;
}

void Fill(ggml_tensor* t, float base, float step) {
  float* d = static_cast<float*>(t->data);
  const int64_t n = ggml_nelements(t);
  for (int64_t i = 0; i < n; ++i) {
    d[i] = base + step * static_cast<float>(i % 11) - 0.1f;
  }
}

TEST(MoeTest, ShapeFiniteAndInputDependent) {
  const OlmoeConfig cfg = TinyCfg();
  const int n_embd = cfg.n_embd, T = 3;

  ggml_init_params ip = {/*mem_size=*/64 * 1024 * 1024, nullptr, false};
  ggml_context* ctx = ggml_init(ip);

  auto mk = [&](int a, int b, int c, float base, float step) {
    ggml_tensor* t = (c > 0) ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a, b, c)
                    : (b > 0) ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b)
                              : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, a);
    Fill(t, base, step);
    return t;
  };

  MoeWeights w;
  w.ffn_norm = mk(n_embd, 0, 0, 1.0f, 0.0f);
  w.router = mk(n_embd, cfg.n_expert, 0, 0.05f, 0.02f);
  w.gate_exps = mk(n_embd, cfg.n_ff, cfg.n_expert, 0.04f, 0.01f);
  w.up_exps = mk(n_embd, cfg.n_ff, cfg.n_expert, 0.03f, 0.015f);
  w.down_exps = mk(cfg.n_ff, n_embd, cfg.n_expert, 0.05f, 0.008f);

  ggml_tensor* h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T);

  ggml_tensor* out = BuildMoe(ctx, cfg, w, h, /*norm_topk=*/true);
  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, out);

  Fill(h, 0.2f, 0.05f);
  ggml_graph_compute_with_ctx(ctx, gf, 1);

  // 形状。
  ASSERT_EQ(out->ne[0], n_embd);
  ASSERT_EQ(out->ne[1], T);

  // 有限 + 记录。
  std::vector<float> before(n_embd * T);
  const int64_t n = ggml_nelements(out);
  ASSERT_EQ(n, static_cast<int64_t>(n_embd * T));
  for (int64_t i = 0; i < n; ++i) {
    const float o = static_cast<float*>(out->data)[i];
    ASSERT_TRUE(std::isfinite(o)) << "non-finite at " << i;
    before[i] = o;
  }

  // 输入相关:改动输入,输出应改变(健全性)。
  static_cast<float*>(h->data)[0] += 3.0f;
  ggml_graph_compute_with_ctx(ctx, gf, 1);
  bool changed = false;
  for (int64_t i = 0; i < n; ++i) {
    if (std::fabs(static_cast<float*>(out->data)[i] - before[i]) > 1e-6f) {
      changed = true;
    }
  }
  EXPECT_TRUE(changed);

  ggml_free(ctx);
}

}  // namespace
}  // namespace nuthatch
