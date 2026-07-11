#include "src/gguf/gguf_reader.h"

#include <cstdio>
#include <string>

#include "ggml.h"
#include "gguf.h"
#include "gtest/gtest.h"

namespace nuthatch {
namespace {

// 用 ggml/gguf 的写 API 现造一个 tiny GGUF,返回文件路径。
// 自包含:测试自己产出输入,不必提交二进制 fixture。
std::string WriteTinyGguf() {
  const std::string path = std::string(testing::TempDir()) + "/tiny.gguf";

  gguf_context* w = gguf_init_empty();
  gguf_set_val_str(w, "general.architecture", "nuthatch_test");
  gguf_set_val_u32(w, "nuthatch_test.block_count", 3);

  // 两个张量(只写形状,数据全 0)。
  ggml_init_params ip = {/*mem_size=*/16 * 1024 * 1024,
                         /*mem_buffer=*/nullptr, /*no_alloc=*/false};
  ggml_context* ctx = ggml_init(ip);
  ggml_tensor* embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
  ggml_set_name(embd, "token_embd.weight");
  ggml_tensor* norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
  ggml_set_name(norm, "output_norm.weight");
  gguf_add_tensor(w, embd);
  gguf_add_tensor(w, norm);

  gguf_write_to_file(w, path.c_str(), /*only_meta=*/false);
  ggml_free(ctx);
  gguf_free(w);
  return path;
}

TEST(GgufReaderTest, ReadsMetadataAndTensorIndex) {
  const std::string path = WriteTinyGguf();

  auto r = GgufReader::Open(path);
  ASSERT_NE(r, nullptr);

  // 元数据。
  EXPECT_EQ(r->architecture(), "nuthatch_test");
  EXPECT_EQ(r->GetU32("nuthatch_test.block_count"), 3u);
  EXPECT_EQ(r->GetU32("does.not.exist", 42u), 42u);  // 缺失 → default

  // 张量索引。
  ASSERT_EQ(r->tensor_count(), 2);
  const auto& t0 = r->tensors()[0];
  EXPECT_EQ(t0.name, "token_embd.weight");
  EXPECT_EQ(t0.type, GGML_TYPE_F32);
  ASSERT_EQ(t0.shape.size(), 2u);
  EXPECT_EQ(t0.shape[0], 4);  // ne[0](最内层)
  EXPECT_EQ(t0.shape[1], 8);

  const auto& t1 = r->tensors()[1];
  EXPECT_EQ(t1.name, "output_norm.weight");
  ASSERT_EQ(t1.shape.size(), 1u);
  EXPECT_EQ(t1.shape[0], 5);

  std::remove(path.c_str());
}

TEST(GgufReaderTest, ReturnsNullOnMissingFile) {
  EXPECT_EQ(GgufReader::Open("/no/such/nuthatch.gguf"), nullptr);
}

}  // namespace
}  // namespace nuthatch
