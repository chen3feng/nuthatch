#include "src/model/moe.h"

#include <cmath>  // INFINITY(Linux/gcc 不会像 macOS 那样间接引入)

namespace nuthatch {

ggml_tensor* BuildMoe(ggml_context* ctx, const OlmoeConfig& cfg,
                      const MoeWeights& w, ggml_tensor* h, bool norm_topk) {
  const int64_t n_embd = cfg.n_embd;
  const int64_t n_expert = cfg.n_expert;
  const int64_t n_used = cfg.n_expert_used;
  const int64_t n_tokens = h->ne[1];

  // post-attention RMSNorm。
  ggml_tensor* b = ggml_mul(ctx, ggml_rms_norm(ctx, h, cfg.rms_eps), w.ffn_norm);

  // 路由:logits → softmax → 概率。
  ggml_tensor* logits = ggml_mul_mat(ctx, w.router, b);  // [n_expert, T]
  ggml_tensor* probs = ggml_soft_max(ctx, logits);       // [n_expert, T]

  // 选 top-k 专家 id。
  ggml_tensor* selected =
      ggml_argsort_top_k(ctx, probs, n_used);  // [n_used, T] (I32)

  // 取选中专家的权重:get_rows 在 [1, n_expert, T] 上按 id 取。
  ggml_tensor* probs3 = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
  ggml_tensor* weights = ggml_get_rows(ctx, probs3, selected);  // [1, n_used, T]

  if (norm_topk) {
    ggml_tensor* w2 = ggml_reshape_2d(ctx, weights, n_used, n_tokens);
    ggml_tensor* wsum = ggml_sum_rows(ctx, w2);              // [1, T]
    wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);  // 防除 0
    w2 = ggml_div(ctx, w2, wsum);
    weights = ggml_reshape_3d(ctx, w2, 1, n_used, n_tokens);
  }

  // 专家 FFN(分离 gate/up/down,SiLU)。
  ggml_tensor* cur = ggml_reshape_3d(ctx, b, n_embd, 1, n_tokens);
  ggml_tensor* up =
      ggml_mul_mat_id(ctx, w.up_exps, cur, selected);  // [n_ff, n_used, T]
  ggml_tensor* gate =
      ggml_mul_mat_id(ctx, w.gate_exps, cur, selected);  // [n_ff, n_used, T]
  gate = ggml_silu(ctx, gate);
  ggml_tensor* ff = ggml_mul(ctx, gate, up);  // [n_ff, n_used, T]
  ggml_tensor* experts =
      ggml_mul_mat_id(ctx, w.down_exps, ff, selected);  // [n_embd, n_used, T]

  // 乘上各专家权重([1, n_used, T] 广播到 n_embd)。
  experts = ggml_mul(ctx, experts, weights);

  // 对 n_used 维求和:逐专家切片相加。
  ggml_tensor* moe_out = nullptr;
  for (int64_t i = 0; i < n_used; ++i) {
    ggml_tensor* slice =
        ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2],
                     i * experts->nb[1]);
    moe_out = (i == 0) ? slice : ggml_add(ctx, moe_out, slice);
  }
  if (n_used == 1) moe_out = ggml_cont(ctx, moe_out);
  return moe_out;  // [n_embd, T]
}

}  // namespace nuthatch
