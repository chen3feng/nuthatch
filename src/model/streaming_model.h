#ifndef NUTHATCH_MODEL_STREAMING_MODEL_H_
#define NUTHATCH_MODEL_STREAMING_MODEL_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "ggml.h"
#include "src/model/expert_reader.h"
#include "src/model/olmoe_model.h"

namespace nuthatch {

// 显存受限加载:把【非专家】权重(注意力/norm/router/embd/lm_head,约占模型
// 一成)常驻;【专家】3D 张量(约九成)不载入内存,交给 ExpertReader 按需流式。
// 这是物理流式执行的前提——OLMoE 常驻从 ~4GB 降到 ~250MB + 槽缓存。
class StreamingModel {
 public:
  // 失败(文件缺失/非法 GGUF)返回 nullptr。
  static std::unique_ptr<StreamingModel> Load(const std::string& path);
  ~StreamingModel();

  StreamingModel(const StreamingModel&) = delete;
  StreamingModel& operator=(const StreamingModel&) = delete;

  const OlmoeConfig& config() const { return cfg_; }

  // 常驻的非专家张量;专家张量(未常驻)或不存在返回 nullptr。
  ggml_tensor* tensor(const std::string& name) const;
  ggml_tensor* layer_tensor(int layer, const std::string& suffix) const;

  // 专家权重的流式读取器(用于把选中的专家装进槽缓存)。
  ExpertReader* reader() const { return reader_.get(); }

 private:
  StreamingModel(ggml_context* ctx, std::unique_ptr<ExpertReader> reader,
                 const OlmoeConfig& cfg,
                 std::unordered_map<std::string, ggml_tensor*> tensors);

  ggml_context* ctx_ = nullptr;  // 仅承载非专家张量的数据
  std::unique_ptr<ExpertReader> reader_;
  OlmoeConfig cfg_;
  std::unordered_map<std::string, ggml_tensor*> tensors_;
};

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_STREAMING_MODEL_H_
