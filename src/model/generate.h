#ifndef NUTHATCH_MODEL_GENERATE_H_
#define NUTHATCH_MODEL_GENERATE_H_

#include <cstdint>
#include <vector>

#include "src/model/olmoe_model.h"

namespace nuthatch {

// 贪心(argmax)生成:对 prompt token ids 反复"整图前向 → 取最后位置 argmax
// → 追加",生成 n_predict 个 token,返回【新生成的】token id(不含 prompt)。
//
// 无 KV cache 版:每步重算整段前向(O(T²);作为 cached 版的正确性基准)。
std::vector<int32_t> GreedyGenerate(const OlmoeModel& model,
                                    std::vector<int32_t> ids, int n_predict,
                                    bool norm_topk);

// 带 KV cache 的贪心生成。第一步 prefill 整段 prompt 填充缓存,之后每步只前向
// 【一个】新 token、注意力对缓存的 K/V 计算(每步 O(1) 计算而非 O(T))。
// 结果应与 GreedyGenerate 逐 token 一致(cached==uncached 是正确性锚点)。
std::vector<int32_t> GreedyGenerateCached(const OlmoeModel& model,
                                          std::vector<int32_t> ids,
                                          int n_predict, bool norm_topk);

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_GENERATE_H_
