#ifndef NUTHATCH_MODEL_STREAMING_FORWARD_H_
#define NUTHATCH_MODEL_STREAMING_FORWARD_H_

#include <cstdint>
#include <vector>

#include "src/model/streaming_model.h"

namespace nuthatch {

// 物理流式贪心生成:专家权重不常驻,推理时每层【按需】把选中的专家从盘装进
// 有界槽缓存(容量 capacity/层)再计算。逐 token 处理,每层两段:
//   段A  注意力(KV cache)+ 残差 + 路由 → 选中的全局专家 id(读回 host)
//   host Ensure 装槽(miss 真 pread)+ 把全局 id 重映射成 slot id
//   段B  在槽张量上 mul_mat_id 算专家 FFN → 残差
// 结果应与常驻路径 GreedyGenerateCached 逐 token 一致(parity),但常驻内存大降。
//
// capacity 须 ≥ n_expert_used。返回【新生成的】token id(不含 prompt)。
std::vector<int32_t> StreamingGenerate(const StreamingModel& model,
                                       std::vector<int32_t> ids, int n_predict,
                                       bool norm_topk, int capacity);

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_STREAMING_FORWARD_H_
