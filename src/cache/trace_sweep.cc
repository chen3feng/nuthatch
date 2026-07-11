#include "src/cache/trace_sweep.h"

#include "src/cache/cache_policy.h"
#include "src/cache/learned_pin_policy.h"
#include "src/cache/lru_policy.h"
#include "src/cache/os_page_cache_policy.h"

namespace nuthatch {

std::vector<SweepRow> SweepBudgets(const RoutingTrace& trace,
                                   const std::vector<uint32_t>& budgets) {
  // learned 需要历史使用直方图(最热专家优先 pin);对整条 trace 统计一次即可。
  const UsageHistogram usage = BuildUsage(trace);

  std::vector<SweepRow> rows;
  rows.reserve(budgets.size());
  for (uint32_t budget : budgets) {
    if (budget == 0) continue;
    const uint32_t pin = budget > 1 ? (budget * 3 + 3) / 4 : 1;  // ≈75% 预留 pin
    const uint32_t lru_slots = budget > pin ? budget - pin : 0;

    LearnedPinPolicy learned(trace.n_layers, pin, lru_slots, usage);
    LruPolicy lru(trace.n_layers, budget);
    OsPageCachePolicy os(static_cast<uint64_t>(budget) * trace.n_layers);

    SweepRow row;
    row.budget = budget;
    row.learned = Replay(trace, &learned).hit_rate();
    row.lru = Replay(trace, &lru).hit_rate();
    row.os = Replay(trace, &os).hit_rate();
    rows.push_back(row);
  }
  return rows;
}

std::vector<PinRatioRow> SweepPinRatio(const RoutingTrace& trace,
                                       uint32_t budget) {
  const UsageHistogram usage = BuildUsage(trace);
  std::vector<PinRatioRow> rows;
  rows.reserve(budget + 1);
  for (uint32_t pin = 0; pin <= budget; ++pin) {
    LearnedPinPolicy learned(trace.n_layers, pin, budget - pin, usage);
    PinRatioRow row;
    row.pin = pin;
    row.lru = budget - pin;
    row.learned = Replay(trace, &learned).hit_rate();
    rows.push_back(row);
  }
  return rows;
}

}  // namespace nuthatch
