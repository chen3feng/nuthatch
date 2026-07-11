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

// 固定总预算下 learned-pin 的 pin/lru 配比曲线。
struct PinRatioRow {
  uint32_t pin;     // pin 常驻的槽数(0..budget)
  uint32_t lru;     // 余下 LRU 槽数(budget - pin)
  double learned;   // 该配比下 learned-pin 命中率
};

// pin 从 0(≈纯 per-layer LRU)扫到 budget(≈全静态最热、无 LRU),固定总预算,
// 看命中率随配比变化——最优往往在中间(pin 抓稳定热点、留几格 LRU 兜临时热点)。
std::vector<PinRatioRow> SweepPinRatio(const RoutingTrace& trace,
                                       uint32_t budget);

}  // namespace nuthatch

#endif  // NUTHATCH_CACHE_TRACE_SWEEP_H_
