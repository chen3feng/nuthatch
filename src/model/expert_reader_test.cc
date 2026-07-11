#include "src/model/expert_reader.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "gtest/gtest.h"

namespace nuthatch {
namespace {

// 写一个含 3D 专家张量 [4,3,5] 的 GGUF,值各不相同(便于逐字节比对)。
std::string WriteFixture() {
  const std::string path = std::string(testing::TempDir()) + "/experts.gguf";
  gguf_context* w = gguf_init_empty();
  ggml_init_params ip = {16 * 1024 * 1024, nullptr, false};
  ggml_context* ctx = ggml_init(ip);

  // 先放一个普通 2D 张量,让后面的专家张量有【非零】文件偏移(验证偏移计算)。
  ggml_tensor* n = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 7, 3);
  ggml_set_name(n, "blk.0.attn_q.weight");
  std::memset(n->data, 0, ggml_nbytes(n));
  gguf_add_tensor(w, n);

  ggml_tensor* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 3, 5);
  ggml_set_name(t, "blk.0.ffn_gate_exps.weight");
  float* d = static_cast<float*>(t->data);
  for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = 0.5f * i - 3.0f;
  gguf_add_tensor(w, t);

  gguf_write_to_file(w, path.c_str(), false);
  ggml_free(ctx);
  gguf_free(w);
  return path;
}

TEST(ExpertReaderTest, ReadsExpertSliceBytes) {
  const std::string path = WriteFixture();
  const std::string name = "blk.0.ffn_gate_exps.weight";

  auto rd = ExpertReader::Open(path);
  ASSERT_NE(rd, nullptr);
  EXPECT_EQ(rd->NumExperts(name), 5);
  const size_t eb = rd->ExpertBytes(name);
  EXPECT_EQ(eb, 4u * 3u * sizeof(float));  // nb[2] = ne0*ne1*4 = 48

  // 常驻对照:把整张张量读进内存,按 nb[2] 切片。
  ggml_context* rc = nullptr;
  gguf_init_params rp = {/*no_alloc=*/false, /*ctx=*/&rc};
  gguf_context* rg = gguf_init_from_file(path.c_str(), rp);
  ASSERT_NE(rg, nullptr);
  ggml_tensor* rt = ggml_get_tensor(rc, name.c_str());
  ASSERT_NE(rt, nullptr);

  for (int e = 0; e < 5; ++e) {
    std::vector<char> buf(eb);
    ASSERT_TRUE(rd->ReadExpert(name, e, buf.data())) << "expert " << e;
    const char* slice = static_cast<const char*>(rt->data) + e * rt->nb[2];
    EXPECT_EQ(std::memcmp(buf.data(), slice, eb), 0) << "expert " << e;
  }

  // 越界 / 不存在的张量。
  std::vector<char> buf(eb);
  EXPECT_FALSE(rd->ReadExpert(name, 5, buf.data()));
  EXPECT_FALSE(rd->ReadExpert(name, -1, buf.data()));
  EXPECT_EQ(rd->ExpertBytes("nope.weight"), 0u);
  EXPECT_EQ(rd->NumExperts("nope.weight"), 0);

  gguf_free(rg);
  ggml_free(rc);
  std::remove(path.c_str());
}

TEST(ExpertReaderTest, OpenFailsOnMissingFile) {
  EXPECT_EQ(ExpertReader::Open("/no/such/file.gguf"), nullptr);
}

}  // namespace
}  // namespace nuthatch
