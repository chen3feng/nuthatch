#include "src/model/forward.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "gtest/gtest.h"
#include "gguf.h"
#include "src/model/olmoe_model.h"

namespace nuthatch {
namespace {

// 造一个结构完整、维度极小的 OLMoE GGUF(2 层)。
std::string WriteTinyOlmoe() {
  const std::string path = std::string(testing::TempDir()) + "/fwd_olmoe.gguf";
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
  int seed = 1;
  auto fill = [&](ggml_tensor* t) {
    float* d = static_cast<float*>(t->data);
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
      seed = seed * 1103515245 + 12345;
      d[i] = 0.02f * static_cast<float>((seed >> 16) % 100 - 50);
    }
  };
  auto t1 = [&](const std::string& n, int a) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, a);
    ggml_set_name(t, n.c_str());
    fill(t);
    gguf_add_tensor(w, t);
  };
  auto t2 = [&](const std::string& n, int a, int b) {
    ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b);
    ggml_set_name(t, n.c_str());
    fill(t);
    gguf_add_tensor(w, t);
  };
  auto t3 = [&](const std::string& n, int a, int b, int c) {
    ggml_tensor* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a, b, c);
    ggml_set_name(t, n.c_str());
    fill(t);
    gguf_add_tensor(w, t);
  };

  t2("token_embd.weight", n_embd, n_vocab);
  t2("output.weight", n_embd, n_vocab);
  t1("output_norm.weight", n_embd);
  for (int l = 0; l < n_layers; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    t1(b + "attn_norm.weight", n_embd);
    t2(b + "attn_q.weight", n_embd, n_embd);
    t2(b + "attn_k.weight", n_embd, n_embd);
    t2(b + "attn_v.weight", n_embd, n_embd);
    t2(b + "attn_output.weight", n_embd, n_embd);
    t1(b + "attn_q_norm.weight", n_embd);
    t1(b + "attn_k_norm.weight", n_embd);
    t1(b + "ffn_norm.weight", n_embd);
    t2(b + "ffn_gate_inp.weight", n_embd, n_expert);
    t3(b + "ffn_gate_exps.weight", n_embd, n_ff, n_expert);
    t3(b + "ffn_up_exps.weight", n_embd, n_ff, n_expert);
    t3(b + "ffn_down_exps.weight", n_ff, n_embd, n_expert);
  }
  gguf_write_to_file(w, path.c_str(), false);
  ggml_free(ctx);
  gguf_free(w);
  return path;
}

TEST(ForwardTest, ShapeFiniteCausal) {
  const std::string path = WriteTinyOlmoe();
  auto model = OlmoeModel::Load(path);
  ASSERT_NE(model, nullptr);
  const int n_vocab = model->config().n_vocab;
  ASSERT_EQ(n_vocab, 16);
  const int T = 4;

  ggml_init_params ip = {128 * 1024 * 1024, nullptr, false};
  ggml_context* ctx = ggml_init(ip);

  ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
  ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);

  ggml_tensor* logits = BuildForward(ctx, *model, ids, pos, /*norm_topk=*/true);
  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, logits);

  int32_t seq[T] = {3, 7, 1, 4};
  for (int i = 0; i < T; ++i) {
    static_cast<int32_t*>(ids->data)[i] = seq[i];
    static_cast<int32_t*>(pos->data)[i] = i;
  }
  ggml_graph_compute_with_ctx(ctx, gf, 1);

  // 形状 [n_vocab, T]。
  ASSERT_EQ(logits->ne[0], n_vocab);
  ASSERT_EQ(logits->ne[1], T);

  // logits(位置 0)有限,记录。
  std::vector<float> l0(n_vocab);
  for (int i = 0; i < n_vocab; ++i) {
    const float x = static_cast<float*>(logits->data)[0 * n_vocab + i];
    ASSERT_TRUE(std::isfinite(x)) << "non-finite logit i=" << i;
    l0[i] = x;
  }

  // 端到端因果:改动最后一个 token 的 id,位置 0 的 logits 应不变。
  static_cast<int32_t*>(ids->data)[T - 1] = 9;
  ggml_graph_compute_with_ctx(ctx, gf, 1);
  for (int i = 0; i < n_vocab; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float*>(logits->data)[0 * n_vocab + i], l0[i])
        << "causality broken at logit i=" << i;
  }

  ggml_free(ctx);
  std::remove(path.c_str());
}

}  // namespace
}  // namespace nuthatch
