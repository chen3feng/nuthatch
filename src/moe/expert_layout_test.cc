#include "src/moe/expert_layout.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "gtest/gtest.h"
#include "src/gguf/gguf_reader.h"
#include "src/io/tensor_source.h"

namespace nuthatch {
namespace {

constexpr int kNEmbd = 4;
constexpr int kNFf = 8;
constexpr int kNExpert = 3;

// 造一个含【融合专家张量】的 tiny GGUF:每个专家填入可区分的值。
std::string WriteMoeGguf() {
  const std::string path = std::string(testing::TempDir()) + "/moe.gguf";
  gguf_context* w = gguf_init_empty();
  gguf_set_val_str(w, "general.architecture", "nuthatch_test");

  ggml_init_params ip = {/*mem_size=*/16 * 1024 * 1024,
                         /*mem_buffer=*/nullptr, /*no_alloc=*/false};
  ggml_context* ctx = ggml_init(ip);

  // 融合专家张量 [n_embd, n_ff, n_expert]。
  ggml_tensor* exps =
      ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kNEmbd, kNFf, kNExpert);
  ggml_set_name(exps, "blk.0.ffn_gate_exps.weight");
  float* d = static_cast<float*>(exps->data);
  for (int e = 0; e < kNExpert; ++e) {
    for (int i = 0; i < kNEmbd * kNFf; ++i) {
      d[e * kNEmbd * kNFf + i] = e * 1000.0f + i;  // 专家 e 可辨识
    }
  }
  gguf_add_tensor(w, exps);

  // 一个非专家张量,确认不会被误识别。
  ggml_tensor* norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
  ggml_set_name(norm, "blk.0.ffn_norm.weight");
  gguf_add_tensor(w, norm);

  gguf_write_to_file(w, path.c_str(), /*only_meta=*/false);
  ggml_free(ctx);
  gguf_free(w);
  return path;
}

TEST(ExpertLayoutTest, IsExpertTensor) {
  EXPECT_TRUE(IsExpertTensor("blk.0.ffn_gate_exps.weight"));
  EXPECT_TRUE(IsExpertTensor("blk.77.ffn_down_exps.weight"));
  EXPECT_FALSE(IsExpertTensor("blk.0.ffn_norm.weight"));
  EXPECT_FALSE(IsExpertTensor("blk.0.ffn_gate_shexp.weight"));  // 共享专家(单数)
  EXPECT_FALSE(IsExpertTensor("token_embd.weight"));
}

TEST(ExpertLayoutTest, ParsesSliceLayoutMatchingGgml) {
  const std::string path = WriteMoeGguf();

  auto reader = GgufReader::Open(path);
  ASSERT_NE(reader, nullptr);
  const std::vector<ExpertTensor> experts = ParseExpertTensors(*reader);

  // 只应识别出那一个融合专家张量。
  ASSERT_EQ(experts.size(), 1u);
  const ExpertTensor& et = experts[0];
  EXPECT_EQ(et.name, "blk.0.ffn_gate_exps.weight");
  EXPECT_EQ(et.n_expert, kNExpert);
  const size_t per_expert_bytes = kNEmbd * kNFf * sizeof(float);
  EXPECT_EQ(et.size_of(), per_expert_bytes);
  EXPECT_EQ(et.OffsetOf(0), et.base_offset);
  EXPECT_EQ(et.OffsetOf(2), et.base_offset + 2 * per_expert_bytes);

  // 对每个专家:TensorSource 按切片读出的字节,应与 ggml 全量读该融合张量后
  // 对应的第 e 段逐字节一致。
  ggml_context* meta = nullptr;
  gguf_init_params p = {/*no_alloc=*/false, /*ctx=*/&meta};
  gguf_context* g = gguf_init_from_file(path.c_str(), p);
  ASSERT_NE(g, nullptr);
  const auto* full = static_cast<const uint8_t*>(
      ggml_get_tensor(meta, "blk.0.ffn_gate_exps.weight")->data);

  auto src = TensorSource::Open(path);
  ASSERT_NE(src, nullptr);
  std::vector<uint8_t> slice(et.size_of());
  for (int64_t e = 0; e < et.n_expert; ++e) {
    ASSERT_TRUE(src->ReadAt(et.OffsetOf(e), et.size_of(), slice.data()));
    EXPECT_EQ(std::memcmp(slice.data(), full + e * per_expert_bytes,
                          per_expert_bytes),
              0)
        << "expert " << e << " slice mismatch";
  }

  gguf_free(g);
  ggml_free(meta);
  std::remove(path.c_str());
}

}  // namespace
}  // namespace nuthatch
