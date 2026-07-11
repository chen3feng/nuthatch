#include "src/model/generate.h"

#include "ggml-cpu.h"
#include "ggml.h"
#include "src/model/forward.h"
#include "src/model/kv_cache.h"

namespace nuthatch {

std::vector<int32_t> GreedyGenerate(const OlmoeModel& model,
                                    std::vector<int32_t> ids, int n_predict,
                                    bool norm_topk) {
  const OlmoeConfig& cfg = model.config();
  std::vector<int32_t> out;
  out.reserve(n_predict);

  for (int step = 0; step < n_predict; ++step) {
    const int T = static_cast<int>(ids.size());

    // 每步的中间张量所需内存随 T 增长,按需给足。
    const size_t mem = static_cast<size_t>(128) * 1024 * 1024 +
                       static_cast<size_t>(cfg.n_vocab) * T * 8 +
                       static_cast<size_t>(cfg.n_layers) * cfg.n_ff *
                           (cfg.n_expert_used + 1) * T * 8;
    ggml_init_params ip = {mem, nullptr, false};
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* tid = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor* logits = BuildForward(ctx, model, tid, pos, norm_topk);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);

    for (int i = 0; i < T; ++i) {
      static_cast<int32_t*>(tid->data)[i] = ids[i];
      static_cast<int32_t*>(pos->data)[i] = i;
    }
    ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/4);

    // 最后位置 argmax。
    const int V = static_cast<int>(cfg.n_vocab);
    const float* last = static_cast<float*>(logits->data) + (T - 1) * V;
    int best = 0;
    for (int i = 1; i < V; ++i) {
      if (last[i] > last[best]) best = i;
    }

    out.push_back(best);
    ids.push_back(best);
    ggml_free(ctx);
  }
  return out;
}

std::vector<int32_t> GreedyGenerateCached(const OlmoeModel& model,
                                          std::vector<int32_t> ids,
                                          int n_predict, bool norm_topk) {
  const OlmoeConfig& cfg = model.config();
  std::vector<int32_t> out;
  out.reserve(n_predict);

  KvCache kv(cfg, static_cast<int>(ids.size()) + n_predict + 1);
  std::vector<int32_t> cur = ids;  // 首步=整段 prompt,之后=单个新 token

  for (int step = 0; step < n_predict; ++step) {
    const int T = static_cast<int>(cur.size());
    const int base = kv.n_past();  // 本步 token 的起始绝对位置

    // decode 步 T=1 内存很小;prefill 步 T=prompt 长度。
    const size_t mem = static_cast<size_t>(128) * 1024 * 1024 +
                       static_cast<size_t>(cfg.n_vocab) * T * 8 +
                       static_cast<size_t>(cfg.n_layers) * cfg.n_ff *
                           (cfg.n_expert_used + 1) * T * 8;
    ggml_init_params ip = {mem, nullptr, false};
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* tid = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    std::vector<ggml_tensor*> cache_writes;
    ggml_tensor* logits =
        BuildForwardCached(ctx, model, kv, tid, pos, norm_topk, &cache_writes);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);
    // cpy-写缓存不在 logits 路径上,需单独 expand 才会执行。
    for (ggml_tensor* w : cache_writes) ggml_build_forward_expand(gf, w);

    for (int i = 0; i < T; ++i) {
      static_cast<int32_t*>(tid->data)[i] = cur[i];
      static_cast<int32_t*>(pos->data)[i] = base + i;
    }
    ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/4);

    const int V = static_cast<int>(cfg.n_vocab);
    const float* last = static_cast<float*>(logits->data) + (T - 1) * V;
    int best = 0;
    for (int i = 1; i < V; ++i) {
      if (last[i] > last[best]) best = i;
    }

    kv.advance(T);        // 本步的 token 已进缓存
    out.push_back(best);
    cur = {best};         // 下一步只需前向这个新 token
    ggml_free(ctx);
  }
  return out;
}

}  // namespace nuthatch
