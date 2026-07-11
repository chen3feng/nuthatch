#include "src/cache/lru_policy.h"

namespace nuthatch {

LruPolicy::LruPolicy(uint32_t n_layers, uint32_t slots_per_layer)
    : slots_(slots_per_layer), layers_(n_layers) {}

bool LruPolicy::Access(uint32_t layer, uint32_t expert) {
  if (layer >= layers_.size()) return false;  // 越界层:当作缺失
  Layer& L = layers_[layer];

  auto it = L.pos.find(expert);
  if (it != L.pos.end()) {
    // 命中:移到队首(标记为最近使用)。
    L.order.splice(L.order.begin(), L.order, it->second);
    return true;
  }

  // 缺失。
  if (slots_ == 0) return false;  // 无缓存,永远缺失
  L.order.push_front(expert);
  L.pos[expert] = L.order.begin();
  if (L.order.size() > slots_) {
    // 超容:驱逐队尾(最久未用)。
    const uint32_t victim = L.order.back();
    L.order.pop_back();
    L.pos.erase(victim);
  }
  return false;
}

}  // namespace nuthatch
