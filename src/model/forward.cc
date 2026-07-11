#include "src/model/forward.h"

#include "src/model/attention.h"
#include "src/model/moe.h"

namespace nuthatch {

ggml_tensor* BuildForward(ggml_context* ctx, const OlmoeModel& model,
                          ggml_tensor* token_ids, ggml_tensor* pos,
                          bool norm_topk) {
  const OlmoeConfig& cfg = model.config();

  // token 嵌入:get_rows 在量化的 token_embd 上按 id 取(反量化成 f32)。
  ggml_tensor* h =
      ggml_get_rows(ctx, model.tensor("token_embd.weight"), token_ids);  // [n_embd, T]

  for (int l = 0; l < static_cast<int>(cfg.n_layers); ++l) {
    // 注意力 + 残差。
    AttnWeights aw;
    aw.attn_norm = model.layer_tensor(l, "attn_norm.weight");
    aw.wq = model.layer_tensor(l, "attn_q.weight");
    aw.wk = model.layer_tensor(l, "attn_k.weight");
    aw.wv = model.layer_tensor(l, "attn_v.weight");
    aw.q_norm = model.layer_tensor(l, "attn_q_norm.weight");
    aw.k_norm = model.layer_tensor(l, "attn_k_norm.weight");
    aw.wo = model.layer_tensor(l, "attn_output.weight");
    h = ggml_add(ctx, h, BuildAttention(ctx, cfg, aw, h, pos));

    // MoE FFN + 残差。
    MoeWeights mw;
    mw.ffn_norm = model.layer_tensor(l, "ffn_norm.weight");
    mw.router = model.layer_tensor(l, "ffn_gate_inp.weight");
    mw.gate_exps = model.layer_tensor(l, "ffn_gate_exps.weight");
    mw.up_exps = model.layer_tensor(l, "ffn_up_exps.weight");
    mw.down_exps = model.layer_tensor(l, "ffn_down_exps.weight");
    h = ggml_add(ctx, h, BuildMoe(ctx, cfg, mw, h, norm_topk));
  }

  // 输出归一 + lm_head。
  h = ggml_mul(ctx, ggml_rms_norm(ctx, h, cfg.rms_eps),
               model.tensor("output_norm.weight"));
  return ggml_mul_mat(ctx, model.tensor("output.weight"), h);  // [n_vocab, T]
}

ggml_tensor* BuildForwardCached(ggml_context* ctx, const OlmoeModel& model,
                                const KvCache& kv, ggml_tensor* token_ids,
                                ggml_tensor* pos, bool norm_topk,
                                std::vector<ggml_tensor*>* cache_writes) {
  const OlmoeConfig& cfg = model.config();
  const int n_past = kv.n_past();

  ggml_tensor* h =
      ggml_get_rows(ctx, model.tensor("token_embd.weight"), token_ids);

  for (int l = 0; l < static_cast<int>(cfg.n_layers); ++l) {
    AttnWeights aw;
    aw.attn_norm = model.layer_tensor(l, "attn_norm.weight");
    aw.wq = model.layer_tensor(l, "attn_q.weight");
    aw.wk = model.layer_tensor(l, "attn_k.weight");
    aw.wv = model.layer_tensor(l, "attn_v.weight");
    aw.q_norm = model.layer_tensor(l, "attn_q_norm.weight");
    aw.k_norm = model.layer_tensor(l, "attn_k_norm.weight");
    aw.wo = model.layer_tensor(l, "attn_output.weight");
    h = ggml_add(ctx, h,
                 BuildAttentionCached(ctx, cfg, aw, h, pos, kv.k(l), kv.v(l),
                                      n_past, cache_writes));

    MoeWeights mw;
    mw.ffn_norm = model.layer_tensor(l, "ffn_norm.weight");
    mw.router = model.layer_tensor(l, "ffn_gate_inp.weight");
    mw.gate_exps = model.layer_tensor(l, "ffn_gate_exps.weight");
    mw.up_exps = model.layer_tensor(l, "ffn_up_exps.weight");
    mw.down_exps = model.layer_tensor(l, "ffn_down_exps.weight");
    h = ggml_add(ctx, h, BuildMoe(ctx, cfg, mw, h, norm_topk));
  }

  h = ggml_mul(ctx, ggml_rms_norm(ctx, h, cfg.rms_eps),
               model.tensor("output_norm.weight"));
  return ggml_mul_mat(ctx, model.tensor("output.weight"), h);
}

}  // namespace nuthatch
