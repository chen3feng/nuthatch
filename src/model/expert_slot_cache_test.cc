#include "src/model/expert_slot_cache.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "gtest/gtest.h"
#include "src/model/expert_reader.h"
#include "src/model/olmoe_model.h"

namespace nuthatch {
namespace {

constexpr int kNEmbd = 8, kNFf = 4, kNExpert = 5;

// 写含一层 gate/up/down 专家张量的 GGUF,每张量每专家值各异(便于比对)。
std::string WriteFixture() {
  const std::string path = std::string(testing::TempDir()) + "/slots.gguf";
  gguf_context* w = gguf_init_empty();
  ggml_init_params ip = {16 * 1024 * 1024, nullptr, false};
  ggml_context* ctx = ggml_init(ip);

  auto add = [&](const std::string& name, int a, int b, float base) {
    ggml_tensor* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a, b, kNExpert);
    ggml_set_name(t, name.c_str());
    float* d = static_cast<float*>(t->data);
    for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = base + 0.25f * i;
    gguf_add_tensor(w, t);
  };
  add("blk.0.ffn_gate_exps.weight", kNEmbd, kNFf, 1.0f);
  add("blk.0.ffn_up_exps.weight", kNEmbd, kNFf, 100.0f);
  add("blk.0.ffn_down_exps.weight", kNFf, kNEmbd, 500.0f);

  gguf_write_to_file(w, path.c_str(), false);
  ggml_free(ctx);
  gguf_free(w);
  return path;
}

OlmoeConfig Cfg() {
  OlmoeConfig c;
  c.n_embd = kNEmbd;
  c.n_ff = kNFf;
  c.n_expert = kNExpert;
  return c;
}

// 校验槽里某专家的 gate 权重字节 == reader 直接读该专家。
void ExpectSlotMatches(const ExpertSlotCache& cache, ExpertReader* rd, int id,
                       int slot) {
  const std::string name = "blk.0.ffn_gate_exps.weight";
  const size_t eb = rd->ExpertBytes(name);
  std::vector<char> want(eb);
  ASSERT_TRUE(rd->ReadExpert(name, id, want.data()));
  const char* got =
      static_cast<const char*>(cache.gate_slots()->data) + slot * cache.gate_slots()->nb[2];
  EXPECT_EQ(std::memcmp(want.data(), got, eb), 0) << "expert " << id;
}

TEST(ExpertSlotCacheTest, LoadHitEvict) {
  const std::string path = WriteFixture();
  auto rd = ExpertReader::Open(path);
  ASSERT_NE(rd, nullptr);

  auto cache = ExpertSlotCache::Create(Cfg(), /*layer=*/0, /*capacity=*/2, rd.get());
  ASSERT_NE(cache, nullptr);

  // 首次:两个都 miss,装入并内容正确。
  auto m = cache->Ensure({0, 1});
  EXPECT_EQ(cache->misses(), 2);
  EXPECT_EQ(cache->hits(), 0);
  ExpectSlotMatches(*cache, rd.get(), 0, m[0]);
  ExpectSlotMatches(*cache, rd.get(), 1, m[1]);
  EXPECT_NE(m[0], m[1]);  // 不同专家占不同槽

  // 再要 {0,1}:全命中。
  cache->Ensure({0, 1});
  EXPECT_EQ(cache->hits(), 2);
  EXPECT_EQ(cache->misses(), 2);

  // 命中一次 0,使 1 成为最久未用(下一次淘汰的目标)。
  cache->Ensure({0});
  EXPECT_EQ(cache->hits(), 3);

  // 容量 2、常驻 {0,1};要 {2} → miss + 淘汰 LRU(此刻是 1)。
  auto m2 = cache->Ensure({2});
  EXPECT_EQ(cache->misses(), 3);
  ExpectSlotMatches(*cache, rd.get(), 2, m2[2]);

  // 0 仍在 → 命中;1 已被淘汰 → miss。
  cache->Ensure({0});
  EXPECT_EQ(cache->hits(), 4);
  cache->Ensure({1});
  EXPECT_EQ(cache->misses(), 4);

  std::remove(path.c_str());
}

TEST(ExpertSlotCacheTest, DedupWithinBatchAndBadInputs) {
  const std::string path = WriteFixture();
  auto rd = ExpertReader::Open(path);
  ASSERT_NE(rd, nullptr);

  auto cache = ExpertSlotCache::Create(Cfg(), 0, 3, rd.get());
  ASSERT_NE(cache, nullptr);

  // 同批重复 id 只算一次 miss。
  auto m = cache->Ensure({3, 3, 3});
  EXPECT_EQ(cache->misses(), 1);
  EXPECT_EQ(m.size(), 1u);

  // 非法容量 / 缺专家张量层 → nullptr。
  EXPECT_EQ(ExpertSlotCache::Create(Cfg(), 0, 0, rd.get()), nullptr);
  EXPECT_EQ(ExpertSlotCache::Create(Cfg(), 9, 3, rd.get()), nullptr);

  std::remove(path.c_str());
}

}  // namespace
}  // namespace nuthatch
