#include "src/trace/routing_trace.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace nuthatch {
namespace {

constexpr char kMagic[8] = {'N', 'U', 'T', 'H', 'T', 'R', 'C', '1'};
constexpr uint32_t kVersion = 1;

template <typename T>
void Put(std::ostream& os, T v) {
  os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
template <typename T>
bool Get(std::istream& is, T* v) {
  return static_cast<bool>(is.read(reinterpret_cast<char*>(v), sizeof(*v)));
}

}  // namespace

bool WriteRoutingTrace(const RoutingTrace& trace, const std::string& path) {
  std::ofstream os(path, std::ios::binary);
  if (!os) return false;
  os.write(kMagic, sizeof(kMagic));
  Put<uint32_t>(os, kVersion);
  Put<uint32_t>(os, trace.n_layers);
  Put<uint32_t>(os, trace.n_expert);
  Put<uint64_t>(os, trace.records.size());
  for (const auto& r : trace.records) {
    Put<uint32_t>(os, r.token);
    Put<uint32_t>(os, r.layer);
    Put<uint32_t>(os, static_cast<uint32_t>(r.experts.size()));
    for (uint32_t e : r.experts) Put<uint32_t>(os, e);
  }
  return static_cast<bool>(os);
}

bool ReadRoutingTrace(const std::string& path, RoutingTrace* out) {
  std::ifstream is(path, std::ios::binary);
  if (!is) return false;

  char magic[8];
  if (!is.read(magic, sizeof(magic)) ||
      std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    return false;  // 魔数错
  }
  uint32_t version = 0, n_layers = 0, n_expert = 0;
  uint64_t n_records = 0;
  if (!Get(is, &version) || version != kVersion || !Get(is, &n_layers) ||
      !Get(is, &n_expert) || !Get(is, &n_records)) {
    return false;  // 版本错 / header 截断
  }

  RoutingTrace t;
  t.n_layers = n_layers;
  t.n_expert = n_expert;
  // 防损坏输入的荒谬预分配:仅在计数看起来合理时才 reserve。
  if (n_records < 1'000'000) t.records.reserve(n_records);
  for (uint64_t i = 0; i < n_records; ++i) {
    RoutingRecord r;
    uint32_t n_sel = 0;
    if (!Get(is, &r.token) || !Get(is, &r.layer) || !Get(is, &n_sel)) {
      return false;  // 记录头截断
    }
    // 选中专家数不应超过每层专家总数(n_expert>0 时),否则视为损坏。
    if (n_expert != 0 && n_sel > n_expert) return false;
    r.experts.resize(n_sel);
    for (uint32_t j = 0; j < n_sel; ++j) {
      if (!Get(is, &r.experts[j])) return false;  // 专家列表截断
    }
    t.records.push_back(std::move(r));
  }
  *out = std::move(t);
  return true;
}

bool ReadRoutingTraceText(const std::string& path, RoutingTrace* out) {
  std::ifstream is(path);
  if (!is) return false;

  RoutingTrace t;
  uint32_t max_layer = 0, max_expert = 0;
  uint32_t token = 0;
  std::string line;
  while (std::getline(is, line)) {
    std::istringstream ls(line);
    RoutingRecord r;
    if (!(ls >> r.layer)) continue;  // 空行/坏行跳过
    r.token = token++;
    uint32_t e = 0;
    while (ls >> e) {
      r.experts.push_back(e);
      if (e > max_expert) max_expert = e;
    }
    if (r.experts.empty()) continue;
    if (r.layer > max_layer) max_layer = r.layer;
    t.records.push_back(std::move(r));
  }
  if (t.records.empty()) return false;
  t.n_layers = max_layer + 1;   // 层号从 0 起
  t.n_expert = max_expert + 1;  // 专家号从 0 起
  *out = std::move(t);
  return true;
}

}  // namespace nuthatch
