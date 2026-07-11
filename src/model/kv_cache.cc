#include "src/model/kv_cache.h"

namespace nuthatch {

KvCache::KvCache(const OlmoeConfig& cfg, int max_seq) : max_seq_(max_seq) {
  const int64_t n_embd_kv =
      static_cast<int64_t>(cfg.n_head_kv) * cfg.head_dim();
  const int n_layers = static_cast<int>(cfg.n_layers);

  // 2 张量/层 × [n_embd_kv, max_seq] f32,加张量头开销。
  const size_t bytes = static_cast<size_t>(n_embd_kv) * max_seq * 4;
  const size_t mem = 2 * n_layers * bytes + 2 * n_layers * ggml_tensor_overhead() +
                     1024 * 1024;
  ggml_init_params ip = {mem, nullptr, /*no_alloc=*/false};
  ctx_ = ggml_init(ip);

  k_.resize(n_layers);
  v_.resize(n_layers);
  for (int l = 0; l < n_layers; ++l) {
    k_[l] = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, n_embd_kv, max_seq);
    v_[l] = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, n_embd_kv, max_seq);
  }
}

KvCache::~KvCache() {
  if (ctx_ != nullptr) ggml_free(ctx_);
}

}  // namespace nuthatch
