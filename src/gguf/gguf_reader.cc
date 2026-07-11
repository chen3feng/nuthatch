#include "src/gguf/gguf_reader.h"

#include <utility>

#include "ggml.h"
#include "gguf.h"

namespace nuthatch {

std::unique_ptr<GgufReader> GgufReader::Open(const std::string& path) {
  ggml_context* meta = nullptr;
  // no_alloc=true + ctx:让 gguf 建一个只含张量【形状】(不分配数据)的
  // ggml_context,用来取每个张量的 ne[]。元数据 KV 仍在 gguf_context 里。
  gguf_init_params params = {/*no_alloc=*/true, /*ctx=*/&meta};
  gguf_context* gguf = gguf_init_from_file(path.c_str(), params);
  if (gguf == nullptr) {
    return nullptr;  // 文件不存在 / 非法 GGUF
  }

  std::vector<TensorInfo> tensors;
  const int64_t n = gguf_get_n_tensors(gguf);
  tensors.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    TensorInfo info;
    info.name = gguf_get_tensor_name(gguf, i);
    info.type = gguf_get_tensor_type(gguf, i);
    info.offset = gguf_get_tensor_offset(gguf, i);
    // 形状从 meta context 里同名张量的 ne[] 取(gguf 本身只给 name/type/offset)。
    if (const ggml_tensor* t = ggml_get_tensor(meta, info.name.c_str())) {
      info.shape.assign(t->ne, t->ne + ggml_n_dims(t));
    }
    tensors.push_back(std::move(info));
  }
  return std::unique_ptr<GgufReader>(
      new GgufReader(gguf, meta, std::move(tensors)));
}

GgufReader::GgufReader(gguf_context* gguf, ggml_context* meta,
                       std::vector<TensorInfo> tensors)
    : gguf_(gguf), meta_(meta), tensors_(std::move(tensors)) {}

GgufReader::~GgufReader() {
  if (gguf_ != nullptr) gguf_free(gguf_);
  if (meta_ != nullptr) ggml_free(meta_);
}

std::string GgufReader::GetStr(const std::string& key,
                               const std::string& default_value) const {
  const int64_t id = gguf_find_key(gguf_, key.c_str());
  if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_STRING) {
    return default_value;
  }
  return gguf_get_val_str(gguf_, id);
}

uint32_t GgufReader::GetU32(const std::string& key,
                            uint32_t default_value) const {
  const int64_t id = gguf_find_key(gguf_, key.c_str());
  if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_UINT32) {
    return default_value;
  }
  return gguf_get_val_u32(gguf_, id);
}

std::string GgufReader::architecture() const {
  return GetStr("general.architecture");
}

}  // namespace nuthatch
