#include "src/core/ggml_smoke.h"

#include "gtest/gtest.h"

namespace nuthatch {

// 打通 blade → vcpkg#ggml → ggml 前向计算 这条链路。
TEST(GgmlSmokeTest, Multiplies) {
  EXPECT_FLOAT_EQ(GgmlMulSmoke(3.0f, 4.0f), 12.0f);
  EXPECT_FLOAT_EQ(GgmlMulSmoke(-2.0f, 5.0f), -10.0f);
  EXPECT_FLOAT_EQ(GgmlMulSmoke(0.0f, 7.0f), 0.0f);
}

}  // namespace nuthatch
