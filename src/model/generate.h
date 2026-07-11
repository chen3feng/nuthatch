#ifndef NUTHATCH_MODEL_GENERATE_H_
#define NUTHATCH_MODEL_GENERATE_H_

#include <cstdint>
#include <vector>

#include "src/model/olmoe_model.h"

namespace nuthatch {

// 贪心(argmax)生成:对 prompt token ids 反复"整图前向 → 取最后位置 argmax
// → 追加",生成 n_predict 个 token,返回【新生成的】token id(不含 prompt)。
//
// 暂无 KV cache:每步重算整段前向(短序列验证足够;KV cache 是后续优化)。
std::vector<int32_t> GreedyGenerate(const OlmoeModel& model,
                                    std::vector<int32_t> ids, int n_predict,
                                    bool norm_topk);

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_GENERATE_H_
