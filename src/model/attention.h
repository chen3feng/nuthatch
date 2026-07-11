#ifndef NUTHATCH_MODEL_ATTENTION_H_
#define NUTHATCH_MODEL_ATTENTION_H_

#include <vector>

#include "ggml.h"
#include "src/model/olmoe_model.h"

namespace nuthatch {

// 一层 OLMoE 注意力需要的权重张量(常驻)。
struct AttnWeights {
  ggml_tensor* attn_norm;  // [n_embd]  输入 RMSNorm 权重
  ggml_tensor* wq;         // [n_embd, n_embd]
  ggml_tensor* wk;         // [n_embd, n_embd]
  ggml_tensor* wv;         // [n_embd, n_embd]
  ggml_tensor* q_norm;     // [n_embd]  QK-norm(OLMoE 对整段 n_embd 归一)
  ggml_tensor* k_norm;     // [n_embd]
  ggml_tensor* wo;         // [n_embd, n_embd]  输出投影
};

// 构建 OLMoE 注意力子图(全序列因果,暂无 KV cache)。
//   x   : [n_embd, n_tokens] 输入隐状态
//   pos : [n_tokens] int32 位置(RoPE 用)
// 返回:注意力输出 [n_embd, n_tokens](未加残差,由调用方相加)。
//
// 算子序列镜像 llama.cpp 的 build_attn_mha:QKV → QK-norm → reshape 头 →
// NEOX RoPE → permute → k^T·q 缩放 → 因果 mask → softmax → ·v → 合并 → o 投影。
// 注:数值正确性在 P14(整图对拍 llama.cpp)验证;本块单测只保证结构/形状/因果。
ggml_tensor* BuildAttention(ggml_context* ctx, const OlmoeConfig& cfg,
                            const AttnWeights& w, ggml_tensor* x,
                            ggml_tensor* pos);

// 带 KV cache 的注意力。x 是新 token [n_embd, T],绝对位置从 n_past 起(pos 给出)。
//   cache_k/cache_v : 本层缓存张量 [n_embd_kv, max_seq](常驻,跨步存活)
//   n_past          : 已缓存的 token 数
//   cache_writes    : 追加"把新 k/v 写入 cache[n_past:n_past+T]"的 cpy 节点,
//                     由调用方 ggml_build_forward_expand 进图(它们不在 logits 路径上)
// 读缓存 [0:n_past] 段(旧数据)拼新 k/v → 完整 K/V;因果 mask 用 n_past 偏移。
// 与 BuildAttention 算子完全对应:prefill(n_past=0)结果应逐元素相同。
ggml_tensor* BuildAttentionCached(ggml_context* ctx, const OlmoeConfig& cfg,
                                  const AttnWeights& w, ggml_tensor* x,
                                  ggml_tensor* pos, ggml_tensor* cache_k,
                                  ggml_tensor* cache_v, int n_past,
                                  std::vector<ggml_tensor*>* cache_writes);

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_ATTENTION_H_
