#ifndef NUTHATCH_CACHE_TRACE_SWEEP_H_
#define NUTHATCH_CACHE_TRACE_SWEEP_H_

#include <cstdint>
#include <vector>

#include "src/trace/routing_trace.h"

namespace nuthatch {

// 一个预算档下三策略的命中率(hit_rate ∈ [0,1])。
struct SweepRow {
  uint32_t budget;   // 每层槽数
  double learned;    // learned-pin(≈75% 槽 pin 历史最热,余下 LRU)
  double lru;        // per-layer LRU,每层 budget 槽
  double os;         // OS 全局页缓存,容量 = budget × n_layers
};

// 在一组 budget 上,用同一条 trace、同一总预算重放三策略,返回每档命中率。
// learned 的 pin/lru 配比固定为 pin=ceil(budget*3/4)、lru=budget-pin(budget≥2;
// budget=1 时 pin=1)。这是"隔离变量、只变预算"看曲线的研究工装。
std::vector<SweepRow> SweepBudgets(const RoutingTrace& trace,
                                   const std::vector<uint32_t>& budgets);

}  // namespace nuthatch

#endif  // NUTHATCH_CACHE_TRACE_SWEEP_H_
