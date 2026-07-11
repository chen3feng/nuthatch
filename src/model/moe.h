#ifndef NUTHATCH_MODEL_MOE_H_
#define NUTHATCH_MODEL_MOE_H_

#include "ggml.h"
#include "src/model/olmoe_model.h"

namespace nuthatch {

// 一层 OLMoE MoE FFN 需要的权重(常驻;专家为融合 3D 张量)。
struct MoeWeights {
  ggml_tensor* ffn_norm;   // [n_embd]  post-attention RMSNorm
  ggml_tensor* router;     // [n_embd, n_expert]  ffn_gate_inp
  ggml_tensor* gate_exps;  // [n_embd, n_ff, n_expert]
  ggml_tensor* up_exps;    // [n_embd, n_ff, n_expert]
  ggml_tensor* down_exps;  // [n_ff, n_embd, n_expert]
};

// 构建 OLMoE MoE FFN 子图:输入 h [n_embd, T],返回 MoE 输出 [n_embd, T]
//(未加残差)。算子序列镜像 llama.cpp build_moe_ffn 的 OLMoE 路径:
//   ffn_norm → router → softmax → argsort_top_k(取 n_expert_used)
//   → get_rows 取权重 →(可选 norm 归一)→ 分离 up/gate 的 mul_mat_id
//   → silu(gate)·up → down mul_mat_id → ×权重 → 各专家切片求和。
// norm_topk:是否把 top-k 权重归一到和为 1(OLMoE 的精确取值在 P14 对拍时定)。
// 数值正确性留 P14 整图对拍;本块单测只保证结构/形状/有限。
ggml_tensor* BuildMoe(ggml_context* ctx, const OlmoeConfig& cfg,
                      const MoeWeights& w, ggml_tensor* h, bool norm_topk);

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_MOE_H_
