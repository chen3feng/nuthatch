#include "src/model/olmoe_model.h"

#include <utility>

#include "ggml.h"
#include "gguf.h"

namespace nuthatch {
namespace {

uint32_t ReadU32(gguf_context* g, const std::string& key, uint32_t def) {
  const int64_t id = gguf_find_key(g, key.c_str());
  if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) return def;
  return gguf_get_val_u32(g, id);
}

float ReadF32(gguf_context* g, const std::string& key, float def) {
  const int64_t id = gguf_find_key(g, key.c_str());
  if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) return def;
  return gguf_get_val_f32(g, id);
}

std::string ReadStr(gguf_context* g, const std::string& key) {
  const int64_t id = gguf_find_key(g, key.c_str());
  if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) return "";
  return gguf_get_val_str(g, id);
}

}  // namespace

OlmoeConfig ParseOlmoeConfig(gguf_context* g, ggml_context* tensors) {
  // 用 general.architecture 作为超参 key 前缀(如 "olmoe.block_count")。
  std::string arch = ReadStr(g, "general.architecture");
  if (arch.empty()) arch = "olmoe";
  auto K = [&](const char* s) { return arch + "." + s; };

  OlmoeConfig cfg;
  cfg.n_layers = ReadU32(g, K("block_count"), 0);
  cfg.n_embd = ReadU32(g, K("embedding_length"), 0);
  cfg.n_head = ReadU32(g, K("attention.head_count"), 0);
  cfg.n_head_kv = ReadU32(g, K("attention.head_count_kv"), cfg.n_head);
  cfg.n_ff = ReadU32(g, K("feed_forward_length"), 0);
  cfg.n_expert = ReadU32(g, K("expert_count"), 0);
  cfg.n_expert_used = ReadU32(g, K("expert_used_count"), 0);
  cfg.n_ctx_train = ReadU32(g, K("context_length"), 0);
  cfg.rope_freq_base = ReadF32(g, K("rope.freq_base"), 10000.0f);
  cfg.rms_eps = ReadF32(g, K("attention.layer_norm_rms_epsilon"), 1e-5f);

  // n_vocab 从 output.weight(退化到 token_embd.weight)的 ne[1] 推。
  ggml_tensor* out = ggml_get_tensor(tensors, "output.weight");
  if (out == nullptr) out = ggml_get_tensor(tensors, "token_embd.weight");
  if (out != nullptr) cfg.n_vocab = static_cast<uint32_t>(out->ne[1]);
  return cfg;
}

std::unique_ptr<OlmoeModel> OlmoeModel::Load(const std::string& path) {
  ggml_context* ctx = nullptr;
  // no_alloc=false:把张量数据也加载进 ctx(常驻)。
  gguf_init_params p = {/*no_alloc=*/false, /*ctx=*/&ctx};
  gguf_context* g = gguf_init_from_file(path.c_str(), p);
  if (g == nullptr) return nullptr;

  const OlmoeConfig cfg = ParseOlmoeConfig(g, ctx);

  // name -> tensor 映射(ctx 里的所有张量)。
  std::unordered_map<std::string, ggml_tensor*> tensors;
  for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr;
       t = ggml_get_next_tensor(ctx, t)) {
    tensors[ggml_get_name(t)] = t;
  }

  return std::unique_ptr<OlmoeModel>(
      new OlmoeModel(g, ctx, cfg, std::move(tensors)));
}

OlmoeModel::OlmoeModel(gguf_context* gguf, ggml_context* ctx,
                       const OlmoeConfig& cfg,
                       std::unordered_map<std::string, ggml_tensor*> tensors)
    : gguf_(gguf), ctx_(ctx), cfg_(cfg), tensors_(std::move(tensors)) {}

OlmoeModel::~OlmoeModel() {
  if (gguf_ != nullptr) gguf_free(gguf_);
  if (ctx_ != nullptr) ggml_free(ctx_);
}

ggml_tensor* OlmoeModel::tensor(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : it->second;
}

ggml_tensor* OlmoeModel::layer_tensor(int layer,
                                      const std::string& suffix) const {
  return tensor("blk." + std::to_string(layer) + "." + suffix);
}

}  // namespace nuthatch
