#include "src/model/streaming_forward.h"

#include <cmath>     // INFINITY
#include <cstdlib>   // malloc/free
#include <cstring>   // std::memcpy
#include <memory>
#include <unordered_map>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "src/model/attention.h"
#include "src/model/expert_slot_cache.h"
#include "src/model/kv_cache.h"

namespace nuthatch {
namespace {

// 小图统一用这个内存上限(逐 token、活值极小;final 的 logits 另加)。
constexpr size_t kGraphMem = 64u * 1024 * 1024;

// 复用同一块 compute 缓冲:每 token 有 34 趟小图,原来每趟 ggml_init 都 malloc
// 一块 64MB、用完 free——纯浪费。各小图顺序执行、用完即 ggml_free,故可共用一块;
// ggml 用外部 buffer 时不接管其释放,我们自己在进程/线程退出时归还。
struct Scratch {
  void* buf = nullptr;
  size_t size = 0;
  ~Scratch() { std::free(buf); }
};
thread_local Scratch g_scratch;

ggml_context* NewCtx(size_t extra = 0) {
  const size_t need = kGraphMem + extra;
  if (g_scratch.size < need) {
    std::free(g_scratch.buf);
    g_scratch.buf = std::malloc(need);
    g_scratch.size = need;
  }
  ggml_init_params ip = {g_scratch.size, g_scratch.buf, /*no_alloc=*/false};
  return ggml_init(ip);
}

// 算出 out 并把它读成 host float 向量(out 必须是 f32)。n_threads:T=1 的极小图
// 用 1(免线程池 spin);只有末端 lm_head(n_vocab×n_embd 大乘)值得多线程。
std::vector<float> Compute1(ggml_context* ctx, ggml_tensor* out,
                            const std::vector<ggml_tensor*>& also,
                            int n_threads) {
  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, out);
  for (ggml_tensor* a : also) ggml_build_forward_expand(gf, a);
  ggml_graph_compute_with_ctx(ctx, gf, n_threads);
  std::vector<float> v(ggml_nelements(out));
  std::memcpy(v.data(), out->data, v.size() * sizeof(float));
  return v;
}

// token 嵌入 → h [n_embd](host)。
std::vector<float> Embed(const StreamingModel& m, int32_t token) {
  ggml_context* ctx = NewCtx();
  ggml_tensor* tid = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
  static_cast<int32_t*>(tid->data)[0] = token;
  ggml_tensor* h = ggml_get_rows(ctx, m.tensor("token_embd.weight"), tid);
  std::vector<float> out = Compute1(ctx, h, {}, /*n_threads=*/1);
  ggml_free(ctx);
  return out;
}

struct PhaseA {
  std::vector<float> h_mid;    // [n_embd] 注意力后残差基
  std::vector<float> weights;  // [n_used] 各选中专家的路由权重
  std::vector<int32_t> selected;  // [n_used] 全局专家 id
};

// 段A:注意力(KV cache)+ 残差 + 路由。
PhaseA RunPhaseA(const StreamingModel& m, KvCache* kv, int l,
                 const std::vector<float>& h_host, int pos, bool norm_topk) {
  const OlmoeConfig& cfg = m.config();
  const int64_t n_embd = cfg.n_embd;
  const int64_t n_expert = cfg.n_expert;
  const int64_t n_used = cfg.n_expert_used;
  ggml_context* ctx = NewCtx();

  ggml_tensor* h_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
  std::memcpy(h_in->data, h_host.data(), n_embd * sizeof(float));
  ggml_tensor* posT = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
  static_cast<int32_t*>(posT->data)[0] = pos;

  AttnWeights aw;
  aw.attn_norm = m.layer_tensor(l, "attn_norm.weight");
  aw.wq = m.layer_tensor(l, "attn_q.weight");
  aw.wk = m.layer_tensor(l, "attn_k.weight");
  aw.wv = m.layer_tensor(l, "attn_v.weight");
  aw.q_norm = m.layer_tensor(l, "attn_q_norm.weight");
  aw.k_norm = m.layer_tensor(l, "attn_k_norm.weight");
  aw.wo = m.layer_tensor(l, "attn_output.weight");
  std::vector<ggml_tensor*> cache_writes;
  ggml_tensor* attn = BuildAttentionCached(ctx, cfg, aw, h_in, posT, kv->k(l),
                                           kv->v(l), pos, &cache_writes);
  ggml_tensor* h_mid = ggml_add(ctx, h_in, attn);

  // 路由(= BuildMoe 的路由段,但这里只算到 selected/weights)。
  ggml_tensor* b =
      ggml_mul(ctx, ggml_rms_norm(ctx, h_mid, cfg.rms_eps),
               m.layer_tensor(l, "ffn_norm.weight"));
  ggml_tensor* logits =
      ggml_mul_mat(ctx, m.layer_tensor(l, "ffn_gate_inp.weight"), b);
  ggml_tensor* probs = ggml_soft_max(ctx, logits);
  ggml_tensor* selected = ggml_argsort_top_k(ctx, probs, n_used);
  ggml_tensor* probs3 = ggml_reshape_3d(ctx, probs, 1, n_expert, 1);
  ggml_tensor* weights = ggml_get_rows(ctx, probs3, selected);  // [1, n_used, 1]
  if (norm_topk) {
    ggml_tensor* w2 = ggml_reshape_2d(ctx, weights, n_used, 1);
    ggml_tensor* wsum = ggml_sum_rows(ctx, w2);
    wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
    w2 = ggml_div(ctx, w2, wsum);
    weights = ggml_reshape_3d(ctx, w2, 1, n_used, 1);
  }

  // 一次图算 h_mid / weights / selected,并让 KV cache 写入生效。
  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, h_mid);
  ggml_build_forward_expand(gf, weights);
  ggml_build_forward_expand(gf, selected);
  for (ggml_tensor* cw : cache_writes) ggml_build_forward_expand(gf, cw);
  ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/1);

  PhaseA r;
  r.h_mid.resize(n_embd);
  std::memcpy(r.h_mid.data(), h_mid->data, n_embd * sizeof(float));
  r.weights.resize(n_used);
  std::memcpy(r.weights.data(), weights->data, n_used * sizeof(float));
  r.selected.resize(n_used);
  std::memcpy(r.selected.data(), selected->data, n_used * sizeof(int32_t));
  ggml_free(ctx);
  return r;
}

// 段B:在槽张量上算专家 FFN + 残差 → h_next [n_embd]。
std::vector<float> RunPhaseB(const StreamingModel& m, int l,
                             const ExpertSlotCache& cache, const PhaseA& a,
                             const std::vector<int32_t>& slot_ids) {
  const OlmoeConfig& cfg = m.config();
  const int64_t n_embd = cfg.n_embd;
  const int64_t n_used = cfg.n_expert_used;
  ggml_context* ctx = NewCtx();

  ggml_tensor* h_mid = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
  std::memcpy(h_mid->data, a.h_mid.data(), n_embd * sizeof(float));
  ggml_tensor* sel = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_used);
  std::memcpy(sel->data, slot_ids.data(), n_used * sizeof(int32_t));
  ggml_tensor* wt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, 1);
  std::memcpy(wt->data, a.weights.data(), n_used * sizeof(float));

  // b 从 h_mid 重算(省一次 host 往返);专家 FFN 同 BuildMoe,但张量用槽 + slot id。
  ggml_tensor* b = ggml_mul(ctx, ggml_rms_norm(ctx, h_mid, cfg.rms_eps),
                            m.layer_tensor(l, "ffn_norm.weight"));
  ggml_tensor* cur = ggml_reshape_3d(ctx, b, n_embd, 1, 1);
  ggml_tensor* up = ggml_mul_mat_id(ctx, cache.up_slots(), cur, sel);
  ggml_tensor* gate = ggml_mul_mat_id(ctx, cache.gate_slots(), cur, sel);
  gate = ggml_silu(ctx, gate);
  ggml_tensor* ff = ggml_mul(ctx, gate, up);
  ggml_tensor* experts = ggml_mul_mat_id(ctx, cache.down_slots(), ff, sel);
  experts = ggml_mul(ctx, experts, wt);  // [n_embd, n_used, 1]

  ggml_tensor* moe = nullptr;
  for (int64_t i = 0; i < n_used; ++i) {
    ggml_tensor* slice = ggml_view_2d(ctx, experts, n_embd, 1, experts->nb[2],
                                      i * experts->nb[1]);
    moe = (i == 0) ? slice : ggml_add(ctx, moe, slice);
  }
  if (n_used == 1) moe = ggml_cont(ctx, moe);
  ggml_tensor* h_next = ggml_add(ctx, h_mid, moe);

  std::vector<float> out = Compute1(ctx, h_next, {}, /*n_threads=*/1);
  ggml_free(ctx);
  return out;
}

// 末端:output_norm + lm_head → logits [n_vocab]。
std::vector<float> RunFinal(const StreamingModel& m,
                            const std::vector<float>& h_host) {
  const OlmoeConfig& cfg = m.config();
  const int64_t n_embd = cfg.n_embd;
  ggml_context* ctx = NewCtx(static_cast<size_t>(cfg.n_vocab) * 8);
  ggml_tensor* h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
  std::memcpy(h->data, h_host.data(), n_embd * sizeof(float));
  ggml_tensor* x = ggml_mul(ctx, ggml_rms_norm(ctx, h, cfg.rms_eps),
                            m.tensor("output_norm.weight"));
  ggml_tensor* logits = ggml_mul_mat(ctx, m.tensor("output.weight"), x);
  std::vector<float> out = Compute1(ctx, logits, {}, /*n_threads=*/4);
  ggml_free(ctx);
  return out;
}

// 处理一个 token,返回其 logits [n_vocab];推进 KV cache。
std::vector<float> Step(const StreamingModel& m, KvCache* kv,
                        std::vector<std::unique_ptr<ExpertSlotCache>>& caches,
                        int32_t token, bool norm_topk) {
  const OlmoeConfig& cfg = m.config();
  const int n_layers = static_cast<int>(cfg.n_layers);
  const int n_used = static_cast<int>(cfg.n_expert_used);
  const int pos = kv->n_past();

  std::vector<float> h = Embed(m, token);
  for (int l = 0; l < n_layers; ++l) {
    PhaseA a = RunPhaseA(m, kv, l, h, pos, norm_topk);
    // 装槽(miss 真从盘 pread)+ 全局 id → slot id。
    std::vector<int> ids(a.selected.begin(), a.selected.end());
    std::unordered_map<int, int> slotmap = caches[l]->Ensure(ids);
    std::vector<int32_t> slot_ids(n_used);
    for (int i = 0; i < n_used; ++i) slot_ids[i] = slotmap[a.selected[i]];
    h = RunPhaseB(m, l, *caches[l], a, slot_ids);
  }
  kv->advance(1);
  return RunFinal(m, h);
}

int Argmax(const std::vector<float>& v) {
  int best = 0;
  for (int i = 1; i < static_cast<int>(v.size()); ++i) {
    if (v[i] > v[best]) best = i;
  }
  return best;
}

}  // namespace

std::vector<int32_t> StreamingGenerate(const StreamingModel& model,
                                       std::vector<int32_t> ids, int n_predict,
                                       bool norm_topk, int capacity) {
  const OlmoeConfig& cfg = model.config();
  std::vector<int32_t> out;
  if (ids.empty() || n_predict <= 0) return out;
  out.reserve(n_predict);

  KvCache kv(cfg, static_cast<int>(ids.size()) + n_predict + 1);
  std::vector<std::unique_ptr<ExpertSlotCache>> caches(cfg.n_layers);
  for (int l = 0; l < static_cast<int>(cfg.n_layers); ++l) {
    caches[l] = ExpertSlotCache::Create(cfg, l, capacity, model.reader());
    if (caches[l] == nullptr) return out;
  }

  // prefill:逐 token 喂完 prompt,留下最后一个的 logits(= 首个生成的预测)。
  std::vector<float> logits;
  for (int32_t t : ids) logits = Step(model, &kv, caches, t, norm_topk);

  for (int step = 0; step < n_predict; ++step) {
    const int32_t next = Argmax(logits);
    out.push_back(next);
    if (step + 1 < n_predict) logits = Step(model, &kv, caches, next, norm_topk);
  }
  return out;
}

}  // namespace nuthatch
