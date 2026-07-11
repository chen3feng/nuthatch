#ifndef NUTHATCH_CACHE_OS_PAGE_CACHE_POLICY_H_
#define NUTHATCH_CACHE_OS_PAGE_CACHE_POLICY_H_

#include <cstdint>
#include <list>
#include <unordered_map>

#include "src/cache/cache_policy.h"

namespace nuthatch {

// OS 页缓存基线:把所有 (layer, expert) 放进【一个全局 LRU】,总容量 capacity。
// 模拟 "llama.cpp + mmap + 靠 OS 页缓存":OS 不知道"层"的概念,只按全局最近
// 使用来驱逐页。用它与 per-layer LRU(相同【总】预算)对比,回答一个研究问题:
// 把缓存"按层分区"到底帮不帮忙?这是 nuthatch 的学习缓存首先要打败的基线。
class OsPageCachePolicy : public CachePolicy {
 public:
  explicit OsPageCachePolicy(uint64_t capacity);

  bool Access(uint32_t layer, uint32_t expert) override;

 private:
  // (layer, expert) 合成一个全局 key;层与专家一起区分条目。
  static uint64_t Key(uint32_t layer, uint32_t expert) {
    return (static_cast<uint64_t>(layer) << 32) | expert;
  }

  uint64_t cap_;
  std::list<uint64_t> order_;  // 前=最近使用,后=最久未用
  std::unordered_map<uint64_t, std::list<uint64_t>::iterator> pos_;
};

}  // namespace nuthatch

#endif  // NUTHATCH_CACHE_OS_PAGE_CACHE_POLICY_H_
