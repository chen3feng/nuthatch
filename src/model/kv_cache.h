#ifndef NUTHATCH_MODEL_KV_CACHE_H_
#define NUTHATCH_MODEL_KV_CACHE_H_

#include <vector>

#include "ggml.h"
#include "src/model/olmoe_model.h"

namespace nuthatch {

// 单序列 KV cache:每层持有 K/V 缓存张量 [n_embd_kv, max_seq](f32,常驻)。
// 位于自己的持久 ggml_context,跨 decode 步存活;每步把新 token 的 k/v 写进
// [n_past:n_past+T] 段,读 [0:n_past] 段拼成完整 K/V(读写区不相交,无图排序问题)。
class KvCache {
 public:
  KvCache(const OlmoeConfig& cfg, int max_seq);
  ~KvCache();

  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;

  ggml_tensor* k(int layer) const { return k_[layer]; }
  ggml_tensor* v(int layer) const { return v_[layer]; }

  int n_past() const { return n_past_; }
  void advance(int n) { n_past_ += n; }
  int max_seq() const { return max_seq_; }

 private:
  ggml_context* ctx_ = nullptr;
  std::vector<ggml_tensor*> k_;
  std::vector<ggml_tensor*> v_;
  int n_past_ = 0;
  int max_seq_ = 0;
};

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_KV_CACHE_H_
