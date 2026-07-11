#ifndef NUTHATCH_MODEL_FORWARD_H_
#define NUTHATCH_MODEL_FORWARD_H_

#include <vector>

#include "ggml.h"
#include "src/model/kv_cache.h"
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

// 带 KV cache 的前向。只对本步的 token(token_ids/pos)建图,注意力读缓存旧段、
// 写新段(cpy 节点追加进 *cache_writes,调用方需 expand 进图)。位置从 kv->n_past()
// 起。返回本步各位置的 logits [n_vocab, T]。逐层与 BuildForward 对应,
// 故 prefill(n_past=0,喂整段 prompt)结果与 BuildForward 相同。
// selected_out(可选):非空时按层序追加每层 MoE 选中的专家 id 张量
// [n_expert_used, T](I32),供 P21 导出真实路由 trace。
ggml_tensor* BuildForwardCached(ggml_context* ctx, const OlmoeModel& model,
                                const KvCache& kv, ggml_tensor* token_ids,
                                ggml_tensor* pos, bool norm_topk,
                                std::vector<ggml_tensor*>* cache_writes,
                                std::vector<ggml_tensor*>* selected_out = nullptr);

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_FORWARD_H_
