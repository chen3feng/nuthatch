#include "src/cache/cache_policy.h"

namespace nuthatch {

ReplayStats Replay(const RoutingTrace& trace, CachePolicy* policy) {
  ReplayStats s;
  for (const RoutingRecord& rec : trace.records) {
    for (uint32_t e : rec.experts) {
      ++s.accesses;
      if (policy->Access(rec.layer, e)) {
        ++s.hits;
      } else {
        ++s.misses;
      }
    }
  }
  return s;
}

}  // namespace nuthatch
