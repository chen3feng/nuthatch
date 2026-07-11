#ifndef NUTHATCH_MODEL_ATTENTION_H_
#define NUTHATCH_MODEL_ATTENTION_H_

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

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_ATTENTION_H_
