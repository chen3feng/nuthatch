#include "src/model/attention.h"

#include <cmath>

namespace nuthatch {

ggml_tensor* BuildAttention(ggml_context* ctx, const OlmoeConfig& cfg,
                            const AttnWeights& w, ggml_tensor* x,
                            ggml_tensor* pos) {
  const int64_t n_embd = cfg.n_embd;
  const int64_t n_head = cfg.n_head;
  const int64_t n_head_kv = cfg.n_head_kv;
  const int64_t head_dim = cfg.head_dim();
  const int64_t n_tokens = x->ne[1];

  // 输入 RMSNorm(×权重)。
  ggml_tensor* cur =
      ggml_mul(ctx, ggml_rms_norm(ctx, x, cfg.rms_eps), w.attn_norm);

  // QKV 投影。
  ggml_tensor* q = ggml_mul_mat(ctx, w.wq, cur);  // [n_embd, T]
  ggml_tensor* k = ggml_mul_mat(ctx, w.wk, cur);
  ggml_tensor* v = ggml_mul_mat(ctx, w.wv, cur);

  // QK-norm:OLMoE 对整段 n_embd 做 RMSNorm(不是 per-head)。
  q = ggml_mul(ctx, ggml_rms_norm(ctx, q, cfg.rms_eps), w.q_norm);
  k = ggml_mul(ctx, ggml_rms_norm(ctx, k, cfg.rms_eps), w.k_norm);

  // 拆头:[head_dim, n_head, T]。
  q = ggml_reshape_3d(ctx, q, head_dim, n_head, n_tokens);
  k = ggml_reshape_3d(ctx, k, head_dim, n_head_kv, n_tokens);
  v = ggml_reshape_3d(ctx, v, head_dim, n_head_kv, n_tokens);

  // NEOX RoPE(全 head_dim 旋转)。
  q = ggml_rope_ext(ctx, q, pos, /*freq_factors=*/nullptr, head_dim,
                    GGML_ROPE_TYPE_NEOX, cfg.n_ctx_train, cfg.rope_freq_base,
                    /*freq_scale=*/1.0f, /*ext_factor=*/0.0f,
                    /*attn_factor=*/1.0f, /*beta_fast=*/32.0f,
                    /*beta_slow=*/1.0f);
  k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                    cfg.n_ctx_train, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f,
                    1.0f);

  // [head_dim, T, n_head]。
  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  k = ggml_permute(ctx, k, 0, 2, 1, 3);
  v = ggml_permute(ctx, v, 0, 2, 1, 3);

  // 打分 + 缩放 + 因果 mask + softmax。
  ggml_tensor* kq = ggml_mul_mat(ctx, k, q);  // [T_k, T_q, n_head]
  kq = ggml_scale(ctx, kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
  kq = ggml_diag_mask_inf(ctx, kq, /*n_past=*/0);  // 全序列因果
  kq = ggml_soft_max(ctx, kq);

  // 上下文:v 需转置成 [T, head_dim, n_head] 再乘。
  v = ggml_cont(ctx, ggml_transpose(ctx, v));   // [T, head_dim, n_head_kv]
  ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);  // [head_dim, T_q, n_head]
  kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);     // [head_dim, n_head, T]
  cur = ggml_cont_2d(ctx, kqv, n_embd, n_tokens);  // [n_embd, T]

  // 输出投影。
  return ggml_mul_mat(ctx, w.wo, cur);
}

}  // namespace nuthatch
