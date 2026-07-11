#include "src/model/kv_cache.h"

#include "ggml.h"
#include "gtest/gtest.h"
#include "src/model/olmoe_model.h"

namespace nuthatch {
namespace {

OlmoeConfig TinyConfig() {
  OlmoeConfig c;
  c.n_layers = 3;
  c.n_embd = 32;
  c.n_head = 4;
  c.n_head_kv = 4;  // head_dim = 8, n_embd_kv = 32
  return c;
}

TEST(KvCacheTest, ShapesAndAdvance) {
  const OlmoeConfig cfg = TinyConfig();
  KvCache kv(cfg, /*max_seq=*/64);

  EXPECT_EQ(kv.n_past(), 0);
  EXPECT_EQ(kv.max_seq(), 64);

  const int64_t n_embd_kv = cfg.n_head_kv * cfg.head_dim();  // 32
  for (int l = 0; l < static_cast<int>(cfg.n_layers); ++l) {
    ASSERT_NE(kv.k(l), nullptr);
    ASSERT_NE(kv.v(l), nullptr);
    EXPECT_EQ(kv.k(l)->ne[0], n_embd_kv);
    EXPECT_EQ(kv.k(l)->ne[1], 64);
    EXPECT_EQ(kv.v(l)->ne[0], n_embd_kv);
    EXPECT_EQ(kv.v(l)->ne[1], 64);
    // 每层 K、V 是各自独立的张量。
    EXPECT_NE(kv.k(l), kv.v(l));
  }

  kv.advance(5);
  EXPECT_EQ(kv.n_past(), 4 + 1);
  kv.advance(1);
  EXPECT_EQ(kv.n_past(), 6);
}

}  // namespace
}  // namespace nuthatch
