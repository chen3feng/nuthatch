#include "src/model/generate.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "gtest/gtest.h"
#include "src/cache/cache_policy.h"
#include "src/cache/lru_policy.h"
#include "src/model/olmoe_model.h"
#include "src/trace/routing_trace.h"

namespace nuthatch {
namespace {

std::string WriteTinyOlmoe() {
  const std::string path = std::string(testing::TempDir()) + "/gen_olmoe.gguf";
  const int n_embd = 8, n_head = 2, n_ff = 4, n_expert = 4, n_used = 2,
            n_vocab = 16, n_layers = 2;
  gguf_context* w = gguf_init_empty();
  gguf_set_val_str(w, "general.architecture", "olmoe");
  gguf_set_val_u32(w, "olmoe.block_count", n_layers);
  gguf_set_val_u32(w, "olmoe.embedding_length", n_embd);
  gguf_set_val_u32(w, "olmoe.attention.head_count", n_head);
  gguf_set_val_u32(w, "olmoe.attention.head_count_kv", n_head);
  gguf_set_val_u32(w, "olmoe.feed_forward_length", n_ff);
  gguf_set_val_u32(w, "olmoe.expert_count", n_expert);
  gguf_set_val_u32(w, "olmoe.expert_used_count", n_used);
  gguf_set_val_u32(w, "olmoe.context_length", 32);
  gguf_set_val_f32(w, "olmoe.rope.freq_base", 10000.0f);
  gguf_set_val_f32(w, "olmoe.attention.layer_norm_rms_epsilon", 1e-5f);

  ggml_init_params ip = {32 * 1024 * 1024, nullptr, false};
  ggml_context* ctx = ggml_init(ip);
  int seed = 7;
  auto add = [&](const std::string& n, int a, int b, int c) {
    ggml_tensor* t = (c > 0) ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a, b, c)
                    : (b > 0) ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b)
                              : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, a);
    ggml_set_name(t, n.c_str());
    float* d = static_cast<float*>(t->data);
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
      seed = seed * 1103515245 + 12345;
      d[i] = 0.02f * static_cast<float>((seed >> 16) % 100 - 50);
    }
    gguf_add_tensor(w, t);
  };
  add("token_embd.weight", n_embd, n_vocab, 0);
  add("output.weight", n_embd, n_vocab, 0);
  add("output_norm.weight", n_embd, 0, 0);
  for (int l = 0; l < n_layers; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    add(b + "attn_norm.weight", n_embd, 0, 0);
    add(b + "attn_q.weight", n_embd, n_embd, 0);
    add(b + "attn_k.weight", n_embd, n_embd, 0);
    add(b + "attn_v.weight", n_embd, n_embd, 0);
    add(b + "attn_output.weight", n_embd, n_embd, 0);
    add(b + "attn_q_norm.weight", n_embd, 0, 0);
    add(b + "attn_k_norm.weight", n_embd, 0, 0);
    add(b + "ffn_norm.weight", n_embd, 0, 0);
    add(b + "ffn_gate_inp.weight", n_embd, n_expert, 0);
    add(b + "ffn_gate_exps.weight", n_embd, n_ff, n_expert);
    add(b + "ffn_up_exps.weight", n_embd, n_ff, n_expert);
    add(b + "ffn_down_exps.weight", n_ff, n_embd, n_expert);
  }
  gguf_write_to_file(w, path.c_str(), false);
  ggml_free(ctx);
  gguf_free(w);
  return path;
}

TEST(GenerateTest, GreedyDeterministicInVocab) {
  const std::string path = WriteTinyOlmoe();
  auto model = OlmoeModel::Load(path);
  ASSERT_NE(model, nullptr);
  const int n_vocab = model->config().n_vocab;

  std::vector<int32_t> prompt = {3, 7};
  std::vector<int32_t> a = GreedyGenerate(*model, prompt, /*n_predict=*/4, true);
  ASSERT_EQ(a.size(), 4u);
  for (int32_t t : a) {
    EXPECT_GE(t, 0);
    EXPECT_LT(t, n_vocab);
  }

  // 贪心 + 确定性:同样输入再来一次,结果一致。
  std::vector<int32_t> b = GreedyGenerate(*model, prompt, 4, true);
  EXPECT_EQ(a, b);

  std::remove(path.c_str());
}

// KV cache 正确性锚点:带缓存生成必须与不带缓存逐 token 一致。
TEST(GenerateTest, CachedMatchesUncached) {
  const std::string path = WriteTinyOlmoe();
  auto model = OlmoeModel::Load(path);
  ASSERT_NE(model, nullptr);

  std::vector<int32_t> prompt = {3, 7, 2, 5};
  for (bool norm_topk : {false, true}) {
    std::vector<int32_t> ref = GreedyGenerate(*model, prompt, /*n_predict=*/6, norm_topk);
    std::vector<int32_t> cached =
        GreedyGenerateCached(*model, prompt, 6, norm_topk);
    EXPECT_EQ(ref, cached) << "norm_topk=" << norm_topk;
  }

  std::remove(path.c_str());
}

// 真实推理路由 trace:良构、可被 M2 缓存策略重放,且不改变生成结果。
TEST(GenerateTest, CachedTraceWellFormedAndReplayable) {
  const std::string path = WriteTinyOlmoe();
  auto model = OlmoeModel::Load(path);
  ASSERT_NE(model, nullptr);
  const auto& cfg = model->config();

  std::vector<int32_t> prompt = {3, 7, 2};
  RoutingTrace tr;
  std::vector<int32_t> gen =
      GreedyGenerateCachedTrace(*model, prompt, /*n_predict=*/5, false, &tr);

  // 带 trace 的生成与不带的逐 token 一致(trace 是旁路,不影响数值)。
  EXPECT_EQ(gen, GreedyGenerateCached(*model, prompt, 5, false));

  EXPECT_EQ(tr.n_layers, cfg.n_layers);
  EXPECT_EQ(tr.n_expert, cfg.n_expert);
  ASSERT_FALSE(tr.records.empty());
  EXPECT_EQ(tr.records.size() % cfg.n_layers, 0u);  // 每 token 一整层栈

  for (const RoutingRecord& r : tr.records) {
    EXPECT_EQ(r.experts.size(), cfg.n_expert_used);
    EXPECT_LT(r.layer, tr.n_layers);
    for (uint32_t e : r.experts) EXPECT_LT(e, tr.n_expert);
  }

  // 现成的 M2 策略能直接重放这条真实 trace。
  LruPolicy lru(tr.n_layers, /*slots_per_layer=*/2);
  ReplayStats s = Replay(tr, &lru);
  EXPECT_GT(s.accesses, 0u);
  EXPECT_EQ(s.hits + s.misses, s.accesses);

  std::remove(path.c_str());
}

}  // namespace
}  // namespace nuthatch
