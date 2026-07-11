#ifndef NUTHATCH_CACHE_LEARNED_PIN_POLICY_H_
#define NUTHATCH_CACHE_LEARNED_PIN_POLICY_H_

#include <cstdint>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/cache/cache_policy.h"
#include "src/trace/routing_trace.h"

namespace nuthatch {

// usage 直方图:每层每专家的历史访问次数。这是 colibrì 的 .coli_usage 的模型
// ——"从过往用量学到的热度"。
struct UsageHistogram {
  std::vector<std::unordered_map<uint32_t, uint64_t>> per_layer;
};

// 从一条 trace 统计 usage 直方图(模拟从历史学习)。
UsageHistogram BuildUsage(const RoutingTrace& trace);

// usage-learned pin:每层按历史热度 pin 最热的 pin_slots 个专家(永不驱逐、
// 启动时预载,故恒命中),其余预算走容量 lru_slots 的 per-layer LRU。
// 这是 colibrì "会学习的缓存"的模型,也是 nuthatch 相对 OS 页缓存基线的差异化核心。
class LearnedPinPolicy : public CachePolicy {
 public:
  LearnedPinPolicy(uint32_t n_layers, uint32_t pin_slots, uint32_t lru_slots,
                   const UsageHistogram& usage);

  bool Access(uint32_t layer, uint32_t expert) override;

 private:
  struct Layer {
    std::unordered_set<uint32_t> pinned;  // 历史最热、预载常驻,恒命中
    std::list<uint32_t> order;            // 其余专家的 per-layer LRU
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> pos;
  };

  uint32_t lru_slots_;
  std::vector<Layer> layers_;
};

}  // namespace nuthatch

#endif  // NUTHATCH_CACHE_LEARNED_PIN_POLICY_H_
