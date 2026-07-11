#include "src/moe/expert_layout.h"

#include <utility>

namespace nuthatch {

bool IsExpertTensor(const std::string& name) {
  static constexpr char kSuffix[] = "exps.weight";
  const size_t n = sizeof(kSuffix) - 1;
  return name.size() >= n && name.compare(name.size() - n, n, kSuffix) == 0;
}

std::vector<ExpertTensor> ParseExpertTensors(const GgufReader& reader) {
  std::vector<ExpertTensor> out;
  for (const auto& t : reader.tensors()) {
    if (!IsExpertTensor(t.name)) continue;
    if (t.shape.size() != 3) continue;  // 融合专家张量必是 3D
    const int64_t n_expert = t.shape[2];
    // n_expert 必须整除总字节:否则专家切片不是整块行(与量化 super-block
    // 对齐冲突),视为布局异常跳过。
    if (n_expert <= 0 || t.size % static_cast<size_t>(n_expert) != 0) {
      continue;
    }
    ExpertTensor et;
    et.name = t.name;
    et.type = t.type;
    et.base_offset = reader.data_offset() + t.offset;
    et.n_expert = n_expert;
    et.expert_stride = t.size / static_cast<size_t>(n_expert);
    out.push_back(std::move(et));
  }
  return out;
}

}  // namespace nuthatch
