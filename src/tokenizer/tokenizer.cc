#include "src/tokenizer/tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <climits>
#include <set>
#include <utility>

#include "gguf.h"

namespace nuthatch {
namespace {

// OLMo 预分词正则(= GPT-2)。PCRE2 支持 \p{L}/\p{N} 与前瞻 (?!\S)。
constexpr char kOlmoPre[] =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S))";

// 把码点编码成 UTF-8(byte-level 码点 < 0x800,1~2 字节足够)。
std::string Utf8Encode(uint32_t cp) {
  std::string s;
  if (cp < 0x80) {
    s.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return s;
}

uint32_t Utf8Next(const std::string& s, size_t i, int* len) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  const size_t n = s.size();
  if (c < 0x80) { *len = 1; return c; }
  if ((c >> 5) == 0x6 && i + 1 < n) {
    *len = 2;
    return ((c & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu);
  }
  if ((c >> 4) == 0xE && i + 2 < n) {
    *len = 3;
    return ((c & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu);
  }
  *len = 1;
  return c;
}

// GPT-2 bytes↔unicode:返回 byte->码点 的表(256 项)。
std::vector<uint32_t> ByteToCodepoint() {
  std::vector<int> bs;
  for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
  std::set<int> in(bs.begin(), bs.end());
  std::vector<uint32_t> cp(256, 0);
  for (int b : bs) cp[b] = static_cast<uint32_t>(b);  // 可打印:码点=字节
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (in.find(b) == in.end()) cp[b] = static_cast<uint32_t>(256 + n++);
  }
  return cp;
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::Load(const std::string& path) {
  gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/nullptr};
  gguf_context* g = gguf_init_from_file(path.c_str(), p);
  if (g == nullptr) return nullptr;

  auto read_str_arr = [&](const char* key, std::vector<std::string>* out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_arr_type(g, id) != GGUF_TYPE_STRING) return false;
    const size_t n = gguf_get_arr_n(g, id);
    out->reserve(n);
    for (size_t i = 0; i < n; ++i) out->emplace_back(gguf_get_arr_str(g, id, i));
    return true;
  };

  std::vector<std::string> vocab, merges;
  const bool ok = read_str_arr("tokenizer.ggml.tokens", &vocab);
  read_str_arr("tokenizer.ggml.merges", &merges);  // 缺 merges 也允许(仅解码)
  gguf_free(g);
  if (!ok) return nullptr;

  return std::unique_ptr<Tokenizer>(
      new Tokenizer(std::move(vocab), std::move(merges)));
}

Tokenizer::Tokenizer(std::vector<std::string> vocab,
                     std::vector<std::string> merges)
    : vocab_(std::move(vocab)) {
  // byte↔char 表。
  const std::vector<uint32_t> cp = ByteToCodepoint();
  byte_to_char_.resize(256);
  for (int b = 0; b < 256; ++b) {
    byte_to_char_[b] = Utf8Encode(cp[b]);
    char_to_byte_[cp[b]] = static_cast<uint8_t>(b);
  }
  // token->id。
  for (int32_t i = 0; i < static_cast<int32_t>(vocab_.size()); ++i) {
    token_to_id_.emplace(vocab_[i], i);
  }
  // merge "A B" -> rank(索引即优先级)。
  for (int i = 0; i < static_cast<int>(merges.size()); ++i) {
    merge_rank_.emplace(merges[i], i);
  }
  // 编译预分词正则。
  int err = 0;
  PCRE2_SIZE eoff = 0;
  regex_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(kOlmoPre),
                         PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP, &err,
                         &eoff, nullptr);
}

Tokenizer::~Tokenizer() {
  if (regex_ != nullptr) pcre2_code_free(static_cast<pcre2_code*>(regex_));
}

const std::string& Tokenizer::token(int32_t id) const {
  static const std::string kEmpty;
  if (id < 0 || id >= static_cast<int32_t>(vocab_.size())) return kEmpty;
  return vocab_[id];
}

void Tokenizer::EncodeChunk(const std::string& chunk,
                            std::vector<int32_t>* out) const {
  if (chunk.empty()) return;
  // byte-level 编码:每个字节 → 一个 byte-level 符号。
  std::vector<std::string> syms;
  syms.reserve(chunk.size());
  for (unsigned char b : chunk) syms.push_back(byte_to_char_[b]);

  // BPE:反复合并 rank 最小的相邻对。
  while (syms.size() > 1) {
    int best_i = -1;
    int best_rank = INT_MAX;
    for (size_t i = 0; i + 1 < syms.size(); ++i) {
      auto it = merge_rank_.find(syms[i] + " " + syms[i + 1]);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = static_cast<int>(i);
      }
    }
    if (best_i < 0) break;
    syms[best_i] += syms[best_i + 1];
    syms.erase(syms.begin() + best_i + 1);
  }

  for (const std::string& s : syms) {
    auto it = token_to_id_.find(s);
    if (it != token_to_id_.end()) out->push_back(it->second);
  }
}

std::vector<int32_t> Tokenizer::Encode(const std::string& text) const {
  std::vector<int32_t> ids;
  if (regex_ == nullptr) return ids;
  pcre2_code* re = static_cast<pcre2_code*>(regex_);
  pcre2_match_data* md = pcre2_match_data_create_from_pattern(re, nullptr);

  PCRE2_SIZE off = 0;
  const PCRE2_SPTR subj = reinterpret_cast<PCRE2_SPTR>(text.data());
  while (off < text.size()) {
    const int rc = pcre2_match(re, subj, text.size(), off, 0, md, nullptr);
    if (rc < 0) break;
    const PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
    const PCRE2_SIZE s = ov[0], e = ov[1];
    if (e <= off) {  // 防止零宽/回退死循环
      ++off;
      continue;
    }
    EncodeChunk(text.substr(s, e - s), &ids);
    off = e;
  }
  pcre2_match_data_free(md);
  return ids;
}

std::string Tokenizer::Decode(const std::vector<int32_t>& ids) const {
  std::string out;
  for (int32_t id : ids) {
    if (id < 0 || id >= static_cast<int32_t>(vocab_.size())) continue;
    const std::string& tok = vocab_[id];
    size_t i = 0;
    while (i < tok.size()) {
      int len = 1;
      const uint32_t c = Utf8Next(tok, i, &len);
      i += len;
      auto it = char_to_byte_.find(c);
      if (it != char_to_byte_.end()) out.push_back(static_cast<char>(it->second));
    }
  }
  return out;
}

}  // namespace nuthatch
