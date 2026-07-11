// 验证 PCRE2 链接 + Unicode 属性类 + 前瞻可用(OLMo 预分词两个关键能力)。
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "gtest/gtest.h"

namespace {

TEST(Pcre2Smoke, UnicodeLetterAndLookahead) {
  int err = 0;
  PCRE2_SIZE eoff = 0;
  // \p{L}+ 后紧跟数字(前瞻)——一次验证 Unicode 属性 + lookahead。
  pcre2_code* re = pcre2_compile(
      reinterpret_cast<PCRE2_SPTR>("\\p{L}+(?=\\d)"), PCRE2_ZERO_TERMINATED,
      PCRE2_UTF | PCRE2_UCP, &err, &eoff, nullptr);
  ASSERT_NE(re, nullptr) << "compile failed at " << eoff;

  pcre2_match_data* md = pcre2_match_data_create_from_pattern(re, nullptr);

  // "café9":\p{L}+(含非 ASCII 字母)后跟数字 → 匹配。
  int rc = pcre2_match(re, reinterpret_cast<PCRE2_SPTR>("caf\xC3\xA9""9"), 6, 0,
                       0, md, nullptr);
  EXPECT_GT(rc, 0);

  // "cafe":后无数字 → 前瞻失败 → 不匹配。
  int rc2 =
      pcre2_match(re, reinterpret_cast<PCRE2_SPTR>("cafe"), 4, 0, 0, md, nullptr);
  EXPECT_LT(rc2, 0);

  pcre2_match_data_free(md);
  pcre2_code_free(re);
}

}  // namespace
