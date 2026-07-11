#include "src/cache/learned_pin_policy.h"

#include <algorithm>
#include <utility>

namespace nuthatch {

UsageHistogram BuildUsage(const RoutingTrace& trace) {
  UsageHistogram u;
  u.per_layer.resize(trace.n_layers);
  for (const RoutingRecord& rec : trace.records) {
    if (rec.layer >= trace.n_layers) continue;
    auto& counts = u.per_layer[rec.layer];
    for (uint32_t e : rec.experts) ++counts[e];
  }
  return u;
}

LearnedPinPolicy::LearnedPinPolicy(uint32_t n_layers, uint32_t pin_slots,
                                   uint32_t lru_slots,
                                   const UsageHistogram& usage)
    : lru_slots_(lru_slots), layers_(n_layers) {
  for (uint32_t l = 0; l < n_layers; ++l) {
    if (l >= usage.per_layer.size()) continue;
    // 取该层历史热度最高的 pin_slots 个专家 pin 住。
    std::vector<std::pair<uint32_t, uint64_t>> v(usage.per_layer[l].begin(),
                                                 usage.per_layer[l].end());
    const size_t k = std::min<size_t>(pin_slots, v.size());
    std::partial_sort(
        v.begin(), v.begin() + k, v.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    for (size_t i = 0; i < k; ++i) layers_[l].pinned.insert(v[i].first);
  }
}

bool LearnedPinPolicy::Access(uint32_t layer, uint32_t expert) {
  if (layer >= layers_.size()) return false;
  Layer& L = layers_[layer];

  // pin 住的专家:启动时预载常驻,恒命中(预载成本是一次性启动开销,不计入
  // 逐 token 的 decode 命中率)。
  if (L.pinned.find(expert) != L.pinned.end()) return true;

  // 其余走 per-layer LRU。
  auto it = L.pos.find(expert);
  if (it != L.pos.end()) {
    L.order.splice(L.order.begin(), L.order, it->second);
    return true;
  }
  if (lru_slots_ == 0) return false;
  L.order.push_front(expert);
  L.pos[expert] = L.order.begin();
  if (L.order.size() > lru_slots_) {
    const uint32_t victim = L.order.back();
    L.order.pop_back();
    L.pos.erase(victim);
  }
  return false;
}

}  // namespace nuthatch
