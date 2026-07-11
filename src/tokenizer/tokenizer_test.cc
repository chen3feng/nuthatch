#include "src/tokenizer/tokenizer.h"

#include <cstdio>
#include <string>

#include "gguf.h"
#include "gtest/gtest.h"

namespace nuthatch {
namespace {

TEST(TokenizerTest, DecodesByteLevel) {
  // 词表:byte-level 编码的 token。
  //  "\xC4\xA0" = U+0120 'Ġ' = space;"\xC4\x8A" = U+010A 'Ċ' = newline。
  const char* toks[] = {"\xC4\xA0Paris", ".", "\xC4\x8A", "Hello"};
  const std::string path = std::string(testing::TempDir()) + "/tok.gguf";

  gguf_context* w = gguf_init_empty();
  gguf_set_arr_str(w, "tokenizer.ggml.tokens", toks, 4);
  gguf_write_to_file(w, path.c_str(), /*only_meta=*/true);
  gguf_free(w);

  auto tk = Tokenizer::Load(path);
  ASSERT_NE(tk, nullptr);
  EXPECT_EQ(tk->vocab_size(), 4);

  EXPECT_EQ(tk->Decode({0, 1}), " Paris.");          // Ġ -> 空格
  EXPECT_EQ(tk->Decode({3, 2, 3}), "Hello\nHello");  // Ċ -> 换行
  EXPECT_EQ(tk->Decode({}), "");
  EXPECT_EQ(tk->Decode({99}), "");  // 越界 id 跳过

  std::remove(path.c_str());
}

TEST(TokenizerTest, ReturnsNullWithoutVocab) {
  // 一个没有 tokenizer.ggml.tokens 的 GGUF。
  const std::string path = std::string(testing::TempDir()) + "/novocab.gguf";
  gguf_context* w = gguf_init_empty();
  gguf_set_val_u32(w, "some.key", 1);
  gguf_write_to_file(w, path.c_str(), /*only_meta=*/true);
  gguf_free(w);

  EXPECT_EQ(Tokenizer::Load(path), nullptr);
  EXPECT_EQ(Tokenizer::Load("/no/such.gguf"), nullptr);
  std::remove(path.c_str());
}

}  // namespace
}  // namespace nuthatch
