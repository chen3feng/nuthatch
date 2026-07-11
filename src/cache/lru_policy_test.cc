#include "src/cache/lru_policy.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "src/cache/cache_policy.h"
#include "src/trace/routing_trace.h"

namespace nuthatch {
namespace {

// 便捷:把一串单专家访问构造成单层 trace。
RoutingTrace SingleLayerTrace(std::vector<uint32_t> experts) {
  RoutingTrace t;
  t.n_layers = 1;
  t.n_expert = 64;
  uint32_t tok = 0;
  for (uint32_t e : experts) t.records.push_back({tok++, 0, {e}});
  return t;
}

TEST(LruPolicyTest, EvictsLeastRecentlyUsed) {
  // 容量 2,序列 1,2,3,1:1 被 3 挤掉后重进 → 全缺失。
  LruPolicy lru(/*n_layers=*/1, /*slots=*/2);
  ReplayStats s = Replay(SingleLayerTrace({1, 2, 3, 1}), &lru);
  EXPECT_EQ(s.accesses, 4u);
  EXPECT_EQ(s.hits, 0u);
  EXPECT_EQ(s.misses, 4u);
}

TEST(LruPolicyTest, HitsWhenCapacitySuffices) {
  // 容量 3,序列 1,2,3,1:最后的 1 命中。
  LruPolicy lru(1, 3);
  ReplayStats s = Replay(SingleLayerTrace({1, 2, 3, 1}), &lru);
  EXPECT_EQ(s.hits, 1u);
  EXPECT_EQ(s.misses, 3u);
}

TEST(LruPolicyTest, RecencyKeepsHotExpert) {
  // 容量 2,序列 1,2,1,3,1:
  //  1 miss{1}; 2 miss{2,1}; 1 hit{1,2}; 3 miss 驱逐2 {3,1}; 1 hit{1,3}
  //  → hits=2 misses=3
  LruPolicy lru(1, 2);
  ReplayStats s = Replay(SingleLayerTrace({1, 2, 1, 3, 1}), &lru);
  EXPECT_EQ(s.hits, 2u);
  EXPECT_EQ(s.misses, 3u);
}

TEST(LruPolicyTest, LayersAreIndependent) {
  // 同一 expert id 在不同层各自独立缓存。
  LruPolicy lru(/*n_layers=*/2, /*slots=*/1);
  RoutingTrace t;
  t.n_layers = 2;
  t.n_expert = 64;
  t.records = {{0, 0, {5}}, {0, 1, {5}}, {1, 0, {5}}, {1, 1, {5}}};
  // 层0: 5 miss, 5 hit;层1: 5 miss, 5 hit → hits=2 misses=2
  ReplayStats s = Replay(t, &lru);
  EXPECT_EQ(s.hits, 2u);
  EXPECT_EQ(s.misses, 2u);
}

TEST(LruPolicyTest, MultiExpertRecordCountsEachAccess) {
  // 一条记录里多个专家,每个都算一次访问。
  LruPolicy lru(1, 8);
  RoutingTrace t;
  t.n_layers = 1;
  t.n_expert = 64;
  t.records = {{0, 0, {1, 2, 3}}, {1, 0, {1, 2, 3}}};  // 第二个 token 全命中
  ReplayStats s = Replay(t, &lru);
  EXPECT_EQ(s.accesses, 6u);
  EXPECT_EQ(s.hits, 3u);
  EXPECT_EQ(s.misses, 3u);
}

TEST(LruPolicyTest, ZeroSlotsAlwaysMisses) {
  LruPolicy lru(1, 0);
  ReplayStats s = Replay(SingleLayerTrace({1, 1, 1}), &lru);
  EXPECT_EQ(s.hits, 0u);
  EXPECT_EQ(s.misses, 3u);
  EXPECT_DOUBLE_EQ(s.hit_rate(), 0.0);
}

}  // namespace
}  // namespace nuthatch
