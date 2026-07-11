#ifndef NUTHATCH_TOKENIZER_TOKENIZER_H_
#define NUTHATCH_TOKENIZER_TOKENIZER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nuthatch {

// GPT-2 byte-level BPE 分词器(OLMoE 用 model=gpt2 + pre=olmo)。
//   Encode:文本 → token id(PCRE2 跑 OLMo 预分词正则 → byte-level → BPE 合并)。
//   Decode:token id → UTF-8 文本(byte↔unicode 反查)。
class Tokenizer {
 public:
  // 从 GGUF 读词表 + merges 并编译预分词正则。失败返回 nullptr。
  static std::unique_ptr<Tokenizer> Load(const std::string& path);
  ~Tokenizer();

  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  std::vector<int32_t> Encode(const std::string& text) const;
  std::string Decode(const std::vector<int32_t>& ids) const;

  int64_t vocab_size() const { return static_cast<int64_t>(vocab_.size()); }
  const std::string& token(int32_t id) const;

 private:
  Tokenizer(std::vector<std::string> vocab, std::vector<std::string> merges);

  // 对一个预分词 chunk(原始 UTF-8)做 byte-level 编码 + BPE 合并,追加 id。
  void EncodeChunk(const std::string& chunk, std::vector<int32_t>* out) const;

  std::vector<std::string> vocab_;                       // id -> byte-level 串
  std::unordered_map<uint32_t, uint8_t> char_to_byte_;   // 码点 -> 字节(decode)
  std::vector<std::string> byte_to_char_;                // 字节 -> byte-level 串(encode)
  std::unordered_map<std::string, int32_t> token_to_id_;  // 串 -> id
  std::unordered_map<std::string, int> merge_rank_;       // "A B" -> 优先级 rank
  void* regex_ = nullptr;                                 // pcre2_code*(opaque)
};

}  // namespace nuthatch

#endif  // NUTHATCH_TOKENIZER_TOKENIZER_H_
