#include "src/util/version.h"

#include "gtest/gtest.h"

namespace nuthatch {

// 冒烟测试:打通 blade + vcpkg#gtest + CI 这条链路。
// 断言版本号存在且非空(项目尚在 0.x)。
TEST(VersionTest, NonEmpty) {
  const char* v = Version();
  ASSERT_NE(v, nullptr);
  EXPECT_STRNE(v, "");
}

}  // namespace nuthatch
