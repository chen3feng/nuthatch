#include "src/cache/learned_pin_policy.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "src/cache/cache_policy.h"
#include "src/cache/lru_policy.h"
#include "src/cache/os_page_cache_policy.h"
#include "src/trace/routing_trace.h"

namespace nuthatch {
namespace {

// 单层 trace:稳定热点 {0,1} 反复出现,夹杂逐个不同的冷专家 churn。
// 冷 churn 会把热专家挤出朴素 LRU,而 pin 住热专家的策略不受影响。
RoutingTrace HotColdTrace() {
  RoutingTrace t;
  t.n_layers = 1;
  t.n_expert = 64;
  uint32_t tok = 0, cold = 10;
  for (int r = 0; r < 8; ++r) {
    t.records.push_back({tok++, 0, {0}});         // hot
    t.records.push_back({tok++, 0, {1}});         // hot
    t.records.push_back({tok++, 0, {cold++}});    // cold(distinct)
    t.records.push_back({tok++, 0, {cold++}});    // cold(distinct)
  }
  return t;
}

// ★ 第一张图的结论(断言形式):相同总预算下,usage-learned pin 的命中率
//   高于 per-layer LRU 和 OS 全局 LRU 两个基线。
TEST(LearnedPinPolicyTest, FirstResultLearnedBeatsBaselines) {
  const RoutingTrace t = HotColdTrace();
  const UsageHistogram u = BuildUsage(t);
  const uint32_t kBudget = 3;  // 每层总预算(本例 1 层)

  LearnedPinPolicy learned(/*n_layers=*/1, /*pin=*/2, /*lru=*/1, u);
  LruPolicy lru(/*n_layers=*/1, kBudget);
  OsPageCachePolicy os(kBudget);

  const ReplayStats sl = Replay(t, &learned);
  const ReplayStats sr = Replay(t, &lru);
  const ReplayStats so = Replay(t, &os);

  EXPECT_GT(sl.hits, sr.hits);   // 学习缓存 > per-layer LRU
  EXPECT_GT(sl.hits, so.hits);   // 学习缓存 > OS 全局 LRU
  EXPECT_EQ(sl.hits, 16u);       // 16 次热专家访问全部 pin-命中
}

TEST(LearnedPinPolicyTest, PinnedAlwaysHitIncludingFirstAccess) {
  // pin 的专家是启动预载,首次访问即命中。
  RoutingTrace hot;
  hot.n_layers = 1;
  hot.n_expert = 64;
  hot.records = {{0, 0, {5}}, {1, 0, {5}}};
  UsageHistogram u = BuildUsage(hot);

  LearnedPinPolicy learned(1, /*pin=*/1, /*lru=*/0, u);
  ReplayStats s = Replay(hot, &learned);
  EXPECT_EQ(s.hits, 2u);  // 首次也命中
  EXPECT_EQ(s.misses, 0u);
}

TEST(LearnedPinPolicyTest, PinsHottestOnly) {
  // 5 访问 10 次、6 访问 2 次;pin=1 只 pin 住 5;lru=0 → 6 恒缺失。
  RoutingTrace t;
  t.n_layers = 1;
  t.n_expert = 64;
  for (int i = 0; i < 10; ++i) t.records.push_back({0, 0, {5}});
  for (int i = 0; i < 2; ++i) t.records.push_back({0, 0, {6}});
  UsageHistogram u = BuildUsage(t);

  LearnedPinPolicy learned(1, /*pin=*/1, /*lru=*/0, u);
  // 只重放各一次做判定。
  EXPECT_TRUE(learned.Access(0, 5));   // 5 被 pin
  EXPECT_FALSE(learned.Access(0, 6));  // 6 未 pin 且无 LRU
}

TEST(LearnedPinPolicyTest, PerLayerPinsAreIndependent) {
  // 层0 热专家是 1;层1 热专家是 2。各 pin 各的。
  RoutingTrace t;
  t.n_layers = 2;
  t.n_expert = 64;
  for (int i = 0; i < 5; ++i) t.records.push_back({0, 0, {1}});
  for (int i = 0; i < 5; ++i) t.records.push_back({0, 1, {2}});
  UsageHistogram u = BuildUsage(t);

  LearnedPinPolicy learned(2, /*pin=*/1, /*lru=*/0, u);
  EXPECT_TRUE(learned.Access(0, 1));   // 层0 pin 了 1
  EXPECT_FALSE(learned.Access(0, 2));  // 层0 没 pin 2
  EXPECT_TRUE(learned.Access(1, 2));   // 层1 pin 了 2
  EXPECT_FALSE(learned.Access(1, 1));  // 层1 没 pin 1
}

TEST(BuildUsageTest, CountsPerLayerPerExpert) {
  RoutingTrace t;
  t.n_layers = 2;
  t.n_expert = 64;
  t.records = {{0, 0, {1, 1, 2}}, {1, 1, {1}}};  // 注:experts 里同 id 计多次
  UsageHistogram u = BuildUsage(t);
  ASSERT_EQ(u.per_layer.size(), 2u);
  EXPECT_EQ(u.per_layer[0][1], 2u);
  EXPECT_EQ(u.per_layer[0][2], 1u);
  EXPECT_EQ(u.per_layer[1][1], 1u);
}

}  // namespace
}  // namespace nuthatch
