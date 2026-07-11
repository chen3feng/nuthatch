#ifndef NUTHATCH_TOKENIZER_TOKENIZER_H_
#define NUTHATCH_TOKENIZER_TOKENIZER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nuthatch {

// GPT-2 byte-level BPE 分词器(OLMoE 用 model=gpt2)。本 PR 只实现【解码】
// (id → UTF-8 文本);编码器(预分词 + BPE)留后续 PR。
//
// 词表里的 token 是 byte-level 编码的字符串(如 space→'Ġ' U+0120,newline→'Ċ'
// U+010A)。解码 = 把每个 token 的每个码点按 GPT-2 byte↔unicode 反查回字节。
class Tokenizer {
 public:
  // 从 GGUF 读 tokenizer.ggml.tokens 词表。失败返回 nullptr。
  static std::unique_ptr<Tokenizer> Load(const std::string& path);

  // token id 序列 → UTF-8 文本。越界 id 跳过。
  std::string Decode(const std::vector<int32_t>& ids) const;

  int64_t vocab_size() const { return static_cast<int64_t>(vocab_.size()); }
  // 单个 id 的原始(byte-level)token 串,便于调试;越界返回空。
  const std::string& token(int32_t id) const;

 private:
  Tokenizer(std::vector<std::string> vocab);

  std::vector<std::string> vocab_;                     // id -> byte-level 串
  std::unordered_map<uint32_t, uint8_t> char_to_byte_;  // 码点 -> 字节
};

}  // namespace nuthatch

#endif  // NUTHATCH_TOKENIZER_TOKENIZER_H_
