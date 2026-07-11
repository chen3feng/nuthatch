#ifndef NUTHATCH_MOE_EXPERT_LAYOUT_H_
#define NUTHATCH_MOE_EXPERT_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"  // ggml_type
#include "src/gguf/gguf_reader.h"

namespace nuthatch {

// GGUF 把一层里的所有专家存成一个【融合 3D 张量】,最外维 ne[2] 是专家索引:
//   blk.N.ffn_gate_exps.weight  ne=[n_embd, n_ff,   n_expert]
//   blk.N.ffn_up_exps.weight    ne=[n_embd, n_ff,   n_expert]
//   blk.N.ffn_down_exps.weight  ne=[n_ff,   n_embd, n_expert]
// 逐专家流式时只需读第三维的某一片(一个 2D 子张量),而不是整个融合张量。
// 因为张量按行主序连续存放、专家是最外维,第 e 个专家就是连续的一段字节。
struct ExpertTensor {
  std::string name;
  ggml_type type;
  size_t base_offset;    // 文件绝对偏移 = reader.data_offset() + tensor.offset
  int64_t n_expert;      // ne[2]
  size_t expert_stride;  // 每个专家 2D 切片的字节数(= 总字节 / n_expert)

  // 第 e 个专家在文件中的绝对偏移;大小为 size_of()。
  size_t OffsetOf(int64_t e) const {
    return base_offset + static_cast<size_t>(e) * expert_stride;
  }
  size_t size_of() const { return expert_stride; }
};

// name 是否是融合专家张量(...exps.weight)。用复数 "exps" 与共享专家的
// "shexp.weight"(单数)区分开。
bool IsExpertTensor(const std::string& name);

// 从 GGUF 张量索引里解析所有融合专家张量的切片布局。
// 跳过非 3D、或总字节不能被 n_expert 整除(布局异常)的张量。
std::vector<ExpertTensor> ParseExpertTensors(const GgufReader& reader);

}  // namespace nuthatch

#endif  // NUTHATCH_MOE_EXPERT_LAYOUT_H_
