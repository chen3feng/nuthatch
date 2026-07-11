// model-free 曲线工具:读一条路由 trace 文件,扫多个每层预算,比三策略命中率。
// 秒级(不重载模型),用于跨 prompt 稳健性 + budget 扫描曲线。
//   trace_replay <trace.bin> [budget1 budget2 ...]   默认 4 8 12 16 24 32
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/cache/trace_sweep.h"
#include "src/trace/routing_trace.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s trace.bin [budget ...]\n", argv[0]);
    return 1;
  }
  nuthatch::RoutingTrace t;
  if (!nuthatch::ReadRoutingTrace(argv[1], &t)) {
    std::fprintf(stderr, "read trace failed: %s\n", argv[1]);
    return 1;
  }

  std::printf("trace: %zu records, %u layers, %u experts/layer\n",
              t.records.size(), t.n_layers, t.n_expert);

  // pin/lru 配比曲线模式:trace_replay <trace> pin <budget>
  if (argc >= 4 && std::string(argv[2]) == "pin") {
    const uint32_t budget = static_cast<uint32_t>(std::atoi(argv[3]));
    std::printf("\n固定预算 %u/层,pin/lru 配比曲线:\n", budget);
    std::printf("  pin  lru   learned\n");
    for (const nuthatch::PinRatioRow& r : nuthatch::SweepPinRatio(t, budget)) {
      std::printf("%5u %4u   %6.1f%%\n", r.pin, r.lru, 100.0 * r.learned);
    }
    return 0;
  }

  std::vector<uint32_t> budgets;
  for (int i = 2; i < argc; ++i) {
    budgets.push_back(static_cast<uint32_t>(std::atoi(argv[i])));
  }
  if (budgets.empty()) budgets = {4, 8, 12, 16, 24, 32};

  std::printf("\nbudget/L   learned     lru       os      learned-lru\n");
  for (const nuthatch::SweepRow& r : nuthatch::SweepBudgets(t, budgets)) {
    std::printf("%6u    %6.1f%%   %6.1f%%   %6.1f%%    %+.1f pp\n", r.budget,
                100.0 * r.learned, 100.0 * r.lru, 100.0 * r.os,
                100.0 * (r.learned - r.lru));
  }
  return 0;
}
