#ifndef NUTHATCH_CACHE_CACHE_POLICY_H_
#define NUTHATCH_CACHE_CACHE_POLICY_H_

#include <cstdint>

#include "src/trace/routing_trace.h"

namespace nuthatch {

// 重放一条 trace 得到的统计。misses 即"需要从磁盘读专家"的次数。
struct ReplayStats {
  uint64_t accesses = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  double hit_rate() const {
    return accesses ? static_cast<double>(hits) / accesses : 0.0;
  }
};

// 缓存策略接口:模拟"某 (layer, expert) 被访问时,它在不在缓存里"。
// 不同策略(LRU / OS 页缓存基线 / usage-learned pin)实现同一接口,用同一条
// trace 重放对比命中率——这是整个研究"隔离变量、对比策略"的核心手法。
class CachePolicy {
 public:
  virtual ~CachePolicy() = default;

  // 访问 (layer, expert)。返回 true=命中(在缓存),false=缺失(需读磁盘)。
  // 实现内部负责更新缓存状态(插入/驱逐/pin)。
  virtual bool Access(uint32_t layer, uint32_t expert) = 0;
};

// 用某策略重放整条 trace 的所有专家访问,累计命中/缺失。
ReplayStats Replay(const RoutingTrace& trace, CachePolicy* policy);

}  // namespace nuthatch

#endif  // NUTHATCH_CACHE_CACHE_POLICY_H_
