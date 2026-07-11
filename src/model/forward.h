#ifndef NUTHATCH_MODEL_FORWARD_H_
#define NUTHATCH_MODEL_FORWARD_H_

#include "ggml.h"
#include "src/model/olmoe_model.h"

namespace nuthatch {

// 构建 OLMoE 的【整个前向图】:token 嵌入 → n_layers×[注意力+残差,
// MoE+残差] → 输出 RMSNorm → lm_head。返回所有位置的 logits [n_vocab, T]。
//
//   ctx        : 建图用的 ggml_context(no_alloc=false,承载中间张量)。
//   model      : 已加载的权重(其 ctx 里的量化张量,mul_mat/get_rows 原生支持)。
//   token_ids  : [T] int32 输入 token id。
//   pos        : [T] int32 位置(RoPE 用)。
//   norm_topk  : MoE top-k 权重是否归一。【OLMoE 用 false】——对拍 llama.cpp
//                确认:true 时首 token 分歧(called vs Paris),false 时 token-exact。
//
// 数值正确性由整图对拍 llama.cpp 验证(本层 CI 测只保证结构/形状/因果)。
ggml_tensor* BuildForward(ggml_context* ctx, const OlmoeModel& model,
                          ggml_tensor* token_ids, ggml_tensor* pos,
                          bool norm_topk);

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_FORWARD_H_
