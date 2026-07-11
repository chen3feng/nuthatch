#include "src/trace/routing_trace.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace nuthatch {
namespace {

std::string TmpPath(const char* name) {
  return std::string(testing::TempDir()) + "/" + name;
}

TEST(RoutingTraceTest, RoundTrip) {
  RoutingTrace t;
  t.n_layers = 4;
  t.n_expert = 8;
  t.records = {
      {/*token=*/0, /*layer=*/0, /*experts=*/{1, 3, 5}},
      {/*token=*/0, /*layer=*/1, /*experts=*/{2, 7}},
      {/*token=*/1, /*layer=*/0, /*experts=*/{0, 3, 5}},
  };

  const std::string path = TmpPath("rt.trace");
  ASSERT_TRUE(WriteRoutingTrace(t, path));

  RoutingTrace back;
  ASSERT_TRUE(ReadRoutingTrace(path, &back));
  EXPECT_EQ(back.n_layers, 4u);
  EXPECT_EQ(back.n_expert, 8u);
  ASSERT_EQ(back.records.size(), 3u);
  EXPECT_EQ(back.records[0].token, 0u);
  EXPECT_EQ(back.records[0].layer, 0u);
  EXPECT_EQ(back.records[0].experts, (std::vector<uint32_t>{1, 3, 5}));
  EXPECT_EQ(back.records[1].experts, (std::vector<uint32_t>{2, 7}));
  EXPECT_EQ(back.records[2].experts, (std::vector<uint32_t>{0, 3, 5}));

  std::remove(path.c_str());
}

TEST(RoutingTraceTest, RejectsMissingFile) {
  RoutingTrace out;
  EXPECT_FALSE(ReadRoutingTrace("/no/such/nuthatch.trace", &out));
}

TEST(RoutingTraceTest, RejectsBadMagic) {
  const std::string path = TmpPath("bad.trace");
  {
    std::ofstream os(path, std::ios::binary);
    os << "GARBAGE!and more";
  }
  RoutingTrace out;
  EXPECT_FALSE(ReadRoutingTrace(path, &out));
  std::remove(path.c_str());
}

TEST(RoutingTraceTest, RejectsTruncatedBody) {
  // 合法 header 但声称有记录、body 却是空的 —— 读到应失败,不崩。
  const std::string path = TmpPath("trunc.trace");
  {
    RoutingTrace t;
    t.n_layers = 2;
    t.n_expert = 8;
    t.records = {{0, 0, {1, 2}}};
    ASSERT_TRUE(WriteRoutingTrace(t, path));
  }
  // 把文件截断到只剩 header(前 24 字节:magic8+ver4+nlayers4+nexpert4... 不含
  // n_records 后的 body),保留 n_records=1 的声明但删掉记录体。
  std::vector<char> bytes;
  {
    std::ifstream is(path, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(is), {});
  }
  ASSERT_GT(bytes.size(), 28u);
  bytes.resize(28);  // 8 magic + 4 ver + 4 nlayers + 4 nexpert + 8 n_records
  {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    os.write(bytes.data(), bytes.size());
  }

  RoutingTrace out;
  EXPECT_FALSE(ReadRoutingTrace(path, &out));  // 声称有记录但读不到
  std::remove(path.c_str());
}

TEST(RoutingTraceTest, ReadsTextTraceInfersDims) {
  const std::string path = std::string(testing::TempDir()) + "/text.trace";
  {
    std::ofstream os(path);
    os << "0 3 7\n";       // layer 0 选专家 3,7
    os << "1 5 2 9\n";     // layer 1 选 5,2,9
    os << "\n";            // 空行跳过
    os << "0 9 1\n";       // layer 0 又一条
  }
  RoutingTrace t;
  ASSERT_TRUE(ReadRoutingTraceText(path, &t));
  EXPECT_EQ(t.records.size(), 3u);       // 空行不计
  EXPECT_EQ(t.n_layers, 2u);             // max layer 1 + 1
  EXPECT_EQ(t.n_expert, 10u);            // max expert 9 + 1
  EXPECT_EQ(t.records[0].layer, 0u);
  EXPECT_EQ(t.records[1].experts, (std::vector<uint32_t>{5, 2, 9}));
  EXPECT_EQ(t.records[2].layer, 0u);

  EXPECT_FALSE(ReadRoutingTraceText("/no/such.txt", &t));  // 缺文件
  std::remove(path.c_str());
}

}  // namespace
}  // namespace nuthatch
