#ifndef NUTHATCH_GGUF_GGUF_READER_H_
#define NUTHATCH_GGUF_GGUF_READER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"  // ggml_type

struct gguf_context;
struct ggml_context;

namespace nuthatch {

// 只读地解析一个 GGUF 文件的【元数据 + 张量索引】,不加载张量数据。
// 用途:plan / 流式加载先知道"有哪些张量、在文件哪个偏移、多大、什么形状",
// 之后再按需 pread 对应字节区间。
class GgufReader {
 public:
  struct TensorInfo {
    std::string name;
    ggml_type type;
    size_t offset;               // 相对张量数据段起点的字节偏移
    std::vector<int64_t> shape;  // ne[]:各维长度,最内层维度在前
  };

  // 打开并解析。文件不存在或非法 GGUF 时返回 nullptr。
  static std::unique_ptr<GgufReader> Open(const std::string& path);
  ~GgufReader();

  GgufReader(const GgufReader&) = delete;
  GgufReader& operator=(const GgufReader&) = delete;

  // 架构名(general.architecture);缺失返回空串。
  std::string architecture() const;

  // 任意标量元数据读取;key 缺失或类型不符时返回 default_value。
  uint32_t GetU32(const std::string& key, uint32_t default_value = 0) const;
  std::string GetStr(const std::string& key,
                     const std::string& default_value = "") const;

  int64_t tensor_count() const {
    return static_cast<int64_t>(tensors_.size());
  }
  const std::vector<TensorInfo>& tensors() const { return tensors_; }

 private:
  GgufReader(gguf_context* gguf, ggml_context* meta,
             std::vector<TensorInfo> tensors);

  gguf_context* gguf_ = nullptr;
  ggml_context* meta_ = nullptr;  // no_alloc:只承载张量形状,不含数据
  std::vector<TensorInfo> tensors_;
};

}  // namespace nuthatch

#endif  // NUTHATCH_GGUF_GGUF_READER_H_
