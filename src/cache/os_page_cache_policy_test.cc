#include "src/cache/os_page_cache_policy.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "src/cache/cache_policy.h"
#include "src/trace/routing_trace.h"

namespace nuthatch {
namespace {

TEST(OsPageCachePolicyTest, GlobalEvictionAcrossLayers) {
  // 全局容量 2,跨层序列 (l0,e1),(l0,e2),(l1,e3),(l0,e1):
  //  miss{(0,1)}; miss{(0,2),(0,1)}; miss 驱逐(0,1) {(1,3),(0,2)}; (0,1)已被驱逐→miss
  //  → 全 miss(工作集 3 > 容量 2)
  OsPageCachePolicy os(2);
  RoutingTrace t;
  t.n_layers = 2;
  t.n_expert = 64;
  t.records = {{0, 0, {1}}, {1, 0, {2}}, {2, 1, {3}}, {3, 0, {1}}};
  ReplayStats s = Replay(t, &os);
  EXPECT_EQ(s.hits, 0u);
  EXPECT_EQ(s.misses, 4u);
}

TEST(OsPageCachePolicyTest, DistinguishesSameExpertInDifferentLayers) {
  // (l0,e5) 与 (l1,e5) 是不同条目;容量 2 可同时装下,重访命中。
  OsPageCachePolicy os(2);
  RoutingTrace t;
  t.n_layers = 2;
  t.n_expert = 64;
  t.records = {{0, 0, {5}}, {1, 1, {5}}, {2, 0, {5}}, {3, 1, {5}}};
  // miss,(miss),hit,hit → hits=2
  ReplayStats s = Replay(t, &os);
  EXPECT_EQ(s.hits, 2u);
  EXPECT_EQ(s.misses, 2u);
}

TEST(OsPageCachePolicyTest, HitsOnReuseWhenFits) {
  // 容量足够容纳整个工作集时,重访全命中。
  OsPageCachePolicy os(8);
  RoutingTrace t;
  t.n_layers = 1;
  t.n_expert = 64;
  t.records = {{0, 0, {1, 2, 3}}, {1, 0, {1, 2, 3}}};
  ReplayStats s = Replay(t, &os);
  EXPECT_EQ(s.hits, 3u);   // 第二个 token 三个专家全命中
  EXPECT_EQ(s.misses, 3u);
}

TEST(OsPageCachePolicyTest, ZeroCapacityAlwaysMisses) {
  OsPageCachePolicy os(0);
  RoutingTrace t;
  t.n_layers = 1;
  t.n_expert = 64;
  t.records = {{0, 0, {7}}, {1, 0, {7}}, {2, 0, {7}}};
  ReplayStats s = Replay(t, &os);
  EXPECT_EQ(s.hits, 0u);
  EXPECT_EQ(s.misses, 3u);
}

}  // namespace
}  // namespace nuthatch
