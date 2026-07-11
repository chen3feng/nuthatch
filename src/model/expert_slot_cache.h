#ifndef NUTHATCH_MODEL_EXPERT_SLOT_CACHE_H_
#define NUTHATCH_MODEL_EXPERT_SLOT_CACHE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "src/model/expert_reader.h"
#include "src/model/olmoe_model.h"

namespace nuthatch {

// 一层 MoE 的【有界专家槽缓存】:C 个槽,每槽驻一个专家的 gate/up/down 三类
// 权重(三类一起进出)。槽底层是 3 个 ggml 张量 [.,.,C],直接作为 mul_mat_id
// 的专家张量;Ensure(选中的专家 ids) 保证它们都在槽里(miss→从 reader 读入、
// 按 LRU 淘汰),返回 专家id→槽号 映射,调用方据此把 top-k id 重映射成 slot id。
//
// 前置:distinct(ids) ≤ capacity。流式前向逐 token、每层选 n_expert_used 个,
// 故 capacity ≥ n_expert_used 即可满足。P20 用 LRU;P22 换 learned-pin 策略。
class ExpertSlotCache {
 public:
  // reader 须在本对象生命周期内有效。张量缺失/容量非法时返回 nullptr。
  static std::unique_ptr<ExpertSlotCache> Create(const OlmoeConfig& cfg,
                                                 int layer, int capacity,
                                                 ExpertReader* reader);
  ~ExpertSlotCache();

  ExpertSlotCache(const ExpertSlotCache&) = delete;
  ExpertSlotCache& operator=(const ExpertSlotCache&) = delete;

  // 保证这批专家都常驻,返回 专家id→槽号。同一批内自动去重。
  std::unordered_map<int, int> Ensure(const std::vector<int>& expert_ids);

  ggml_tensor* gate_slots() const { return gate_; }  // [n_embd, n_ff, C]
  ggml_tensor* up_slots() const { return up_; }      // [n_embd, n_ff, C]
  ggml_tensor* down_slots() const { return down_; }  // [n_ff, n_embd, C]

  int capacity() const { return capacity_; }
  int64_t hits() const { return hits_; }
  int64_t misses() const { return misses_; }

 private:
  ExpertSlotCache(ExpertReader* reader, ggml_context* ctx, int capacity,
                  std::string gate, std::string up, std::string down);
  int AcquireSlot();                        // 空槽优先,否则 LRU 淘汰
  bool LoadSlot(int slot, int expert_id);   // 读 gate/up/down 进该槽

  ExpertReader* reader_;
  ggml_context* ctx_;
  ggml_tensor* gate_ = nullptr;
  ggml_tensor* up_ = nullptr;
  ggml_tensor* down_ = nullptr;
  std::string gate_name_, up_name_, down_name_;
  int capacity_;
  std::unordered_map<int, int> id_to_slot_;  // 专家id → 槽
  std::vector<int> slot_to_id_;              // 槽 → 专家id(-1 空)
  std::vector<uint64_t> slot_tick_;          // 槽最近使用 tick(LRU)
  uint64_t tick_ = 0;
  int64_t hits_ = 0;
  int64_t misses_ = 0;
};

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_EXPERT_SLOT_CACHE_H_
