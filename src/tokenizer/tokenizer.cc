#include "src/tokenizer/tokenizer.h"

#include <set>
#include <utility>

#include "gguf.h"

namespace nuthatch {
namespace {

// GPT-2 bytes↔unicode:可打印字节映射到自身码点,其余顺序映射到 256+n。
// 返回 码点->字节 的反查表。
std::unordered_map<uint32_t, uint8_t> BuildCharToByte() {
  std::vector<int> bs;
  for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
  std::set<int> in(bs.begin(), bs.end());
  std::vector<int> cs = bs;  // 对应码点(可打印字节:码点=字节值)
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (in.find(b) == in.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  std::unordered_map<uint32_t, uint8_t> m;
  for (size_t i = 0; i < bs.size(); ++i) {
    m[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
  }
  return m;
}

// 读 s[i..] 的一个 UTF-8 码点;*len 为字节数(1..4)。越界/非法按单字节处理。
uint32_t Utf8Next(const std::string& s, size_t i, int* len) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  const size_t n = s.size();
  if (c < 0x80) {
    *len = 1;
    return c;
  }
  if ((c >> 5) == 0x6 && i + 1 < n) {
    *len = 2;
    return ((c & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu);
  }
  if ((c >> 4) == 0xE && i + 2 < n) {
    *len = 3;
    return ((c & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu);
  }
  if ((c >> 3) == 0x1E && i + 3 < n) {
    *len = 4;
    return ((c & 0x07u) << 18) | ((s[i + 1] & 0x3Fu) << 12) |
           ((s[i + 2] & 0x3Fu) << 6) | (s[i + 3] & 0x3Fu);
  }
  *len = 1;
  return c;
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::Load(const std::string& path) {
  gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/nullptr};
  gguf_context* g = gguf_init_from_file(path.c_str(), p);
  if (g == nullptr) return nullptr;

  const int64_t id = gguf_find_key(g, "tokenizer.ggml.tokens");
  if (id < 0 || gguf_get_arr_type(g, id) != GGUF_TYPE_STRING) {
    gguf_free(g);
    return nullptr;
  }
  const size_t n = gguf_get_arr_n(g, id);
  std::vector<std::string> vocab;
  vocab.reserve(n);
  for (size_t i = 0; i < n; ++i) vocab.emplace_back(gguf_get_arr_str(g, id, i));
  gguf_free(g);

  return std::unique_ptr<Tokenizer>(new Tokenizer(std::move(vocab)));
}

Tokenizer::Tokenizer(std::vector<std::string> vocab)
    : vocab_(std::move(vocab)), char_to_byte_(BuildCharToByte()) {}

const std::string& Tokenizer::token(int32_t id) const {
  static const std::string kEmpty;
  if (id < 0 || id >= static_cast<int32_t>(vocab_.size())) return kEmpty;
  return vocab_[id];
}

std::string Tokenizer::Decode(const std::vector<int32_t>& ids) const {
  std::string out;
  for (int32_t id : ids) {
    if (id < 0 || id >= static_cast<int32_t>(vocab_.size())) continue;
    const std::string& tok = vocab_[id];
    size_t i = 0;
    while (i < tok.size()) {
      int len = 1;
      const uint32_t cp = Utf8Next(tok, i, &len);
      i += len;
      auto it = char_to_byte_.find(cp);
      if (it != char_to_byte_.end()) out.push_back(static_cast<char>(it->second));
    }
  }
  return out;
}

}  // namespace nuthatch
