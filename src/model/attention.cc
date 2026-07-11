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

ggml_tensor* BuildAttentionCached(ggml_context* ctx, const OlmoeConfig& cfg,
                                  const AttnWeights& w, ggml_tensor* x,
                                  ggml_tensor* pos, ggml_tensor* cache_k,
                                  ggml_tensor* cache_v, int n_past,
                                  std::vector<ggml_tensor*>* cache_writes) {
  const int64_t n_embd = cfg.n_embd;
  const int64_t n_head = cfg.n_head;
  const int64_t n_head_kv = cfg.n_head_kv;
  const int64_t head_dim = cfg.head_dim();
  const int64_t n_embd_kv = n_head_kv * head_dim;
  const int64_t T = x->ne[1];       // 本步新 token 数
  const int64_t n_kv = n_past + T;  // 注意力可见的 K/V 长度

  // —— 与 BuildAttention 相同的前半段:norm → QKV → QK-norm → 拆头 → RoPE ——
  ggml_tensor* cur =
      ggml_mul(ctx, ggml_rms_norm(ctx, x, cfg.rms_eps), w.attn_norm);
  ggml_tensor* q = ggml_mul_mat(ctx, w.wq, cur);
  ggml_tensor* k = ggml_mul_mat(ctx, w.wk, cur);
  ggml_tensor* v = ggml_mul_mat(ctx, w.wv, cur);  // [n_embd_kv, T],入缓存
  q = ggml_mul(ctx, ggml_rms_norm(ctx, q, cfg.rms_eps), w.q_norm);
  k = ggml_mul(ctx, ggml_rms_norm(ctx, k, cfg.rms_eps), w.k_norm);
  q = ggml_reshape_3d(ctx, q, head_dim, n_head, T);
  k = ggml_reshape_3d(ctx, k, head_dim, n_head_kv, T);
  q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                    cfg.n_ctx_train, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f,
                    1.0f);
  k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                    cfg.n_ctx_train, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f,
                    1.0f);

  // 缓存里存的是 RoPE 之后的 K(按绝对位置旋转过);V 不旋转。都摊平成 [n_embd_kv, T]。
  ggml_tensor* k2 = ggml_reshape_2d(ctx, k, n_embd_kv, T);

  // 把新 k/v 写入 cache[n_past:n_past+T](供下一步读;本步不读这段,读写不相交)。
  const size_t k_off = static_cast<size_t>(n_past) * cache_k->nb[1];
  const size_t v_off = static_cast<size_t>(n_past) * cache_v->nb[1];
  cache_writes->push_back(ggml_cpy(
      ctx, k2, ggml_view_2d(ctx, cache_k, n_embd_kv, T, cache_k->nb[1], k_off)));
  cache_writes->push_back(ggml_cpy(
      ctx, v, ggml_view_2d(ctx, cache_v, n_embd_kv, T, cache_v->nb[1], v_off)));

  // 完整 K/V = 缓存旧段 [0:n_past] 拼本步新 k/v(concat 显式依赖新 k/v,排序天然正确)。
  ggml_tensor* k_full = k2;
  ggml_tensor* v_full = v;
  if (n_past > 0) {
    ggml_tensor* k_old =
        ggml_view_2d(ctx, cache_k, n_embd_kv, n_past, cache_k->nb[1], 0);
    ggml_tensor* v_old =
        ggml_view_2d(ctx, cache_v, n_embd_kv, n_past, cache_v->nb[1], 0);
    k_full = ggml_concat(ctx, k_old, k2, /*dim=*/1);  // [n_embd_kv, n_kv]
    v_full = ggml_concat(ctx, v_old, v, /*dim=*/1);
  }

  // —— 与 BuildAttention 相同的后半段(K/V 长度用 n_kv,mask 偏移 n_past)——
  ggml_tensor* K = ggml_reshape_3d(ctx, k_full, head_dim, n_head_kv, n_kv);
  ggml_tensor* V = ggml_reshape_3d(ctx, v_full, head_dim, n_head_kv, n_kv);
  q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [head_dim, T, n_head]
  K = ggml_permute(ctx, K, 0, 2, 1, 3);  // [head_dim, n_kv, n_head_kv]
  V = ggml_permute(ctx, V, 0, 2, 1, 3);

  ggml_tensor* kq = ggml_mul_mat(ctx, K, q);  // [n_kv, T, n_head]
  kq = ggml_scale(ctx, kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
  kq = ggml_diag_mask_inf(ctx, kq, n_past);  // 偏移因果:query i 见 key ≤ n_past+i
  kq = ggml_soft_max(ctx, kq);

  V = ggml_cont(ctx, ggml_transpose(ctx, V));  // [n_kv, head_dim, n_head_kv]
  ggml_tensor* kqv = ggml_mul_mat(ctx, V, kq);  // [head_dim, T, n_head]
  kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
  cur = ggml_cont_2d(ctx, kqv, n_embd, T);
  return ggml_mul_mat(ctx, w.wo, cur);
}

}  // namespace nuthatch
