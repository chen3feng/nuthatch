#ifndef NUTHATCH_MODEL_OLMOE_MODEL_H_
#define NUTHATCH_MODEL_OLMOE_MODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "ggml.h"

struct gguf_context;

namespace nuthatch {

// OLMoE-1B-7B 架构参数(从 GGUF 元数据读出)。
// 参考(真模型):16 层 · n_embd 2048 · 16 头(MHA,无 GQA)· head_dim 128 ·
// 64 专家选 8 · ff/专家 1024 · RoPE θ=10000 · QK-norm · 无共享专家 · vocab 50304。
struct OlmoeConfig {
  uint32_t n_layers = 0;
  uint32_t n_embd = 0;
  uint32_t n_head = 0;
  uint32_t n_head_kv = 0;
  uint32_t n_ff = 0;           // 每个专家的中间维
  uint32_t n_expert = 0;       // 专家总数
  uint32_t n_expert_used = 0;  // 每 token 选中的专家数(top-k)
  uint32_t n_vocab = 0;
  uint32_t n_ctx_train = 0;
  float rope_freq_base = 10000.0f;
  float rms_eps = 1e-5f;

  uint32_t head_dim() const { return n_head ? n_embd / n_head : 0; }
};

// 从 GGUF 元数据 + 张量形状解析架构参数(常驻 / 流式两种 loader 共用)。
// tensors 是承载(至少形状的)张量的 ctx——n_vocab 由 output.weight 的 ne[1] 推。
OlmoeConfig ParseOlmoeConfig(gguf_context* g, ggml_context* tensors);

// 加载 OLMoE 模型:读 config,并把【全部权重张量常驻】进一个 ggml_context
// (no_alloc=false)。这是"先常驻、后流式"的第一步——先让引擎能跑通前向,
// 之后再把专家换成流式 backend buffer(M3 后半)。
class OlmoeModel {
 public:
  // 失败(文件缺失/非法 GGUF)返回 nullptr。
  static std::unique_ptr<OlmoeModel> Load(const std::string& path);
  ~OlmoeModel();

  OlmoeModel(const OlmoeModel&) = delete;
  OlmoeModel& operator=(const OlmoeModel&) = delete;

  const OlmoeConfig& config() const { return cfg_; }

  // 按名取权重张量;不存在返回 nullptr。
  ggml_tensor* tensor(const std::string& name) const;
  // blk.<layer>.<suffix> 的便捷取法。
  ggml_tensor* layer_tensor(int layer, const std::string& suffix) const;

  // 承载权重的 ctx,供前向图构建复用(后续 PR)。
  ggml_context* weights_ctx() const { return ctx_; }

 private:
  OlmoeModel(gguf_context* gguf, ggml_context* ctx, const OlmoeConfig& cfg,
             std::unordered_map<std::string, ggml_tensor*> tensors);

  gguf_context* gguf_ = nullptr;
  ggml_context* ctx_ = nullptr;  // 持有所有权重张量的数据
  OlmoeConfig cfg_;
  std::unordered_map<std::string, ggml_tensor*> tensors_;
};

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_OLMOE_MODEL_H_
