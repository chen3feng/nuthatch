#ifndef NUTHATCH_CACHE_LRU_POLICY_H_
#define NUTHATCH_CACHE_LRU_POLICY_H_

#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

#include "src/cache/cache_policy.h"

namespace nuthatch {

// 每层独立的 LRU:每层最多缓存 slots_per_layer 个专家,满了驱逐最久未用的。
// 这是 colibrì 的 per-layer 缓存基线,也是 usage-learned 策略要打败的对手之一。
class LruPolicy : public CachePolicy {
 public:
  LruPolicy(uint32_t n_layers, uint32_t slots_per_layer);

  bool Access(uint32_t layer, uint32_t expert) override;

 private:
  struct Layer {
    std::list<uint32_t> order;  // 前=最近使用,后=最久未用
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> pos;
  };

  uint32_t slots_;
  std::vector<Layer> layers_;
};

}  // namespace nuthatch

#endif  // NUTHATCH_CACHE_LRU_POLICY_H_
