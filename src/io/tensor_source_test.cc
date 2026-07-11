#include "src/io/tensor_source.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "gtest/gtest.h"
#include "src/gguf/gguf_reader.h"

namespace nuthatch {
namespace {

// 造一个 tiny GGUF,张量填入已知数据。
std::string WriteTinyGguf() {
  const std::string path = std::string(testing::TempDir()) + "/ts.gguf";
  gguf_context* w = gguf_init_empty();
  gguf_set_val_str(w, "general.architecture", "nuthatch_test");
  ggml_init_params ip = {/*mem_size=*/16 * 1024 * 1024,
                         /*mem_buffer=*/nullptr, /*no_alloc=*/false};
  ggml_context* ctx = ggml_init(ip);
  ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6);
  ggml_set_name(t, "blk.0.weight");
  float* d = static_cast<float*>(t->data);
  for (int i = 0; i < 6; ++i) d[i] = i * 1.5f + 0.25f;
  gguf_add_tensor(w, t);
  gguf_write_to_file(w, path.c_str(), /*only_meta=*/false);
  ggml_free(ctx);
  gguf_free(w);
  return path;
}

// TensorSource 的 pread 应与 ggml 自己读出的张量字节逐字节一致。
TEST(TensorSourceTest, ReadsSameBytesAsGgml) {
  const std::string path = WriteTinyGguf();

  // (a) 让 ggml 自己把数据加载进 ctx(no_alloc=false)。
  ggml_context* meta = nullptr;
  gguf_init_params p = {/*no_alloc=*/false, /*ctx=*/&meta};
  gguf_context* g = gguf_init_from_file(path.c_str(), p);
  ASSERT_NE(g, nullptr);
  ggml_tensor* gt = ggml_get_tensor(meta, "blk.0.weight");
  ASSERT_NE(gt, nullptr);
  const size_t nbytes = ggml_nbytes(gt);

  // (b) nuthatch 路径:GgufReader 给偏移/大小,TensorSource 定位 pread。
  auto reader = GgufReader::Open(path);
  ASSERT_NE(reader, nullptr);
  ASSERT_EQ(reader->tensor_count(), 1);
  const auto& info = reader->tensors()[0];
  EXPECT_EQ(info.size, nbytes);

  auto src = TensorSource::Open(path);
  ASSERT_NE(src, nullptr);
  std::vector<uint8_t> buf(info.size);
  const size_t abs_off = reader->data_offset() + info.offset;
  ASSERT_TRUE(src->ReadAt(abs_off, info.size, buf.data()));

  // (c) 逐字节相等。
  EXPECT_EQ(std::memcmp(buf.data(), gt->data, nbytes), 0);

  // 越界读(超过文件尾)应返回 false。
  std::vector<uint8_t> big(info.size + 4096);
  EXPECT_FALSE(src->ReadAt(abs_off, big.size(), big.data()));

  gguf_free(g);
  ggml_free(meta);
  std::remove(path.c_str());
}

TEST(TensorSourceTest, ReturnsNullOnMissingFile) {
  EXPECT_EQ(TensorSource::Open("/no/such/nuthatch.bin"), nullptr);
}

}  // namespace
}  // namespace nuthatch
