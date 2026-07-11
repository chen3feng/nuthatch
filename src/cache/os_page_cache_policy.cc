#include "src/cache/os_page_cache_policy.h"

namespace nuthatch {

OsPageCachePolicy::OsPageCachePolicy(uint64_t capacity) : cap_(capacity) {}

bool OsPageCachePolicy::Access(uint32_t layer, uint32_t expert) {
  const uint64_t k = Key(layer, expert);

  auto it = pos_.find(k);
  if (it != pos_.end()) {
    order_.splice(order_.begin(), order_, it->second);  // 命中:移到队首
    return true;
  }

  if (cap_ == 0) return false;
  order_.push_front(k);
  pos_[k] = order_.begin();
  if (order_.size() > cap_) {
    const uint64_t victim = order_.back();  // 全局最久未用
    order_.pop_back();
    pos_.erase(victim);
  }
  return false;
}

}  // namespace nuthatch
