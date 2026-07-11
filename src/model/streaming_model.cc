#include "src/model/streaming_model.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <utility>

#include "gguf.h"

namespace nuthatch {
namespace {

// 专家 3D 张量名都含 "_exps.weight"(gate/up/down);router 是 ffn_gate_inp,不含。
bool IsExpertTensor(const char* name) {
  return std::strstr(name, "_exps.weight") != nullptr;
}

}  // namespace

std::unique_ptr<StreamingModel> StreamingModel::Load(const std::string& path) {
  // no_alloc=true:只拿元数据 + 张量形状,不载任何权重数据。
  ggml_context* meta = nullptr;
  gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/&meta};
  gguf_context* g = gguf_init_from_file(path.c_str(), p);
  if (g == nullptr) return nullptr;

  const OlmoeConfig cfg = ParseOlmoeConfig(g, meta);

  // 常驻 ctx 只给非专家张量留空间。
  size_t bytes = 0;
  int count = 0;
  for (ggml_tensor* t = ggml_get_first_tensor(meta); t != nullptr;
       t = ggml_get_next_tensor(meta, t)) {
    if (IsExpertTensor(ggml_get_name(t))) continue;
    bytes += ggml_nbytes(t);
    ++count;
  }
  const size_t mem =
      bytes + static_cast<size_t>(count + 2) * ggml_tensor_overhead() + (1u << 20);
  ggml_init_params rip = {mem, nullptr, /*no_alloc=*/false};
  ggml_context* rctx = ggml_init(rip);

  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    ggml_free(rctx);
    ggml_free(meta);
    gguf_free(g);
    return nullptr;
  }
  const uint64_t data_off = gguf_get_data_offset(g);

  // 逐个非专家张量:在常驻 ctx 建同形张量并从文件 pread 其数据。
  std::unordered_map<std::string, ggml_tensor*> tensors;
  bool ok = true;
  for (ggml_tensor* t = ggml_get_first_tensor(meta); t != nullptr && ok;
       t = ggml_get_next_tensor(meta, t)) {
    const char* name = ggml_get_name(t);
    if (IsExpertTensor(name)) continue;  // 专家流式,不常驻

    ggml_tensor* r = ggml_dup_tensor(rctx, t);  // 同 type/ne,数据在 rctx
    ggml_set_name(r, name);
    const int64_t id = gguf_find_tensor(g, name);
    const uint64_t off = data_off + gguf_get_tensor_offset(g, id);
    const size_t nb = ggml_nbytes(r);
    if (pread(fd, r->data, nb, static_cast<off_t>(off)) !=
        static_cast<ssize_t>(nb)) {
      ok = false;
      break;
    }
    tensors[name] = r;
  }
  close(fd);
  ggml_free(meta);
  gguf_free(g);

  if (!ok) {
    ggml_free(rctx);
    return nullptr;
  }

  auto reader = ExpertReader::Open(path);
  if (reader == nullptr) {
    ggml_free(rctx);
    return nullptr;
  }

  return std::unique_ptr<StreamingModel>(
      new StreamingModel(rctx, std::move(reader), cfg, std::move(tensors)));
}

StreamingModel::StreamingModel(
    ggml_context* ctx, std::unique_ptr<ExpertReader> reader,
    const OlmoeConfig& cfg,
    std::unordered_map<std::string, ggml_tensor*> tensors)
    : ctx_(ctx),
      reader_(std::move(reader)),
      cfg_(cfg),
      tensors_(std::move(tensors)) {}

StreamingModel::~StreamingModel() {
  if (ctx_ != nullptr) ggml_free(ctx_);
}

ggml_tensor* StreamingModel::tensor(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : it->second;
}

ggml_tensor* StreamingModel::layer_tensor(int layer,
                                          const std::string& suffix) const {
  return tensor("blk." + std::to_string(layer) + "." + suffix);
}

}  // namespace nuthatch
