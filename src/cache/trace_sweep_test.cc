#include "src/cache/trace_sweep.h"

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "src/trace/routing_trace.h"

namespace nuthatch {
namespace {

// 单层、偏斜访问:热专家 {0,1} 反复,冷专家 2..15 轮转穿插。
RoutingTrace SkewedTrace() {
  RoutingTrace t;
  t.n_layers = 1;
  t.n_expert = 16;
  uint32_t token = 0;
  for (int round = 0; round < 20; ++round) {
    const uint32_t cold = 2 + (round % 14);
    for (uint32_t e : {0u, 1u, cold}) {
      t.records.push_back({token++, 0, {e}});
    }
  }
  return t;
}

TEST(TraceSweepTest, LearnedHoldsEdgeAndMonotonic) {
  const RoutingTrace t = SkewedTrace();
  const std::vector<uint32_t> budgets = {2, 3, 4, 6};
  const std::vector<SweepRow> rows = SweepBudgets(t, budgets);

  ASSERT_EQ(rows.size(), budgets.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(rows[i].budget, budgets[i]);
    // 命中率合法。
    for (double r : {rows[i].learned, rows[i].lru, rows[i].os}) {
      EXPECT_GE(r, 0.0);
      EXPECT_LE(r, 1.0);
    }
    // 偏斜 trace 上,pin 热专家的 learned 不劣于 LRU/OS。
    EXPECT_GE(rows[i].learned, rows[i].lru - 1e-9);
    EXPECT_GE(rows[i].learned, rows[i].os - 1e-9);
  }
  // 预算越大命中率不降(每策略单调不减)。
  for (size_t i = 1; i < rows.size(); ++i) {
    EXPECT_GE(rows[i].learned, rows[i - 1].learned - 1e-9);
    EXPECT_GE(rows[i].lru, rows[i - 1].lru - 1e-9);
    EXPECT_GE(rows[i].os, rows[i - 1].os - 1e-9);
  }
}

TEST(TraceSweepTest, SkipsZeroBudget) {
  const std::vector<SweepRow> rows = SweepBudgets(SkewedTrace(), {0, 2});
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows[0].budget, 2u);
}

TEST(TraceSweepTest, PinRatioCurveHelpsOnSkew) {
  const uint32_t budget = 4;
  const std::vector<PinRatioRow> rows = SweepPinRatio(SkewedTrace(), budget);

  ASSERT_EQ(rows.size(), budget + 1);  // pin = 0..budget
  double best = 0.0;
  for (uint32_t i = 0; i <= budget; ++i) {
    EXPECT_EQ(rows[i].pin, i);
    EXPECT_EQ(rows[i].pin + rows[i].lru, budget);
    EXPECT_GE(rows[i].learned, 0.0);
    EXPECT_LE(rows[i].learned, 1.0);
    best = std::max(best, rows[i].learned);
  }
  // 偏斜 trace 上,某个 pin>0 的配比不劣于 pin=0(纯 LRU)——pinning 有用。
  EXPECT_GE(best, rows[0].learned - 1e-9);
}

}  // namespace
}  // namespace nuthatch
