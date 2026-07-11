#include "src/model/streaming_forward.h"

#include <cstdio>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "gtest/gtest.h"
#include "src/model/generate.h"
#include "src/model/olmoe_model.h"
#include "src/model/streaming_model.h"

namespace nuthatch {
namespace {

std::string WriteTinyOlmoe() {
  const std::string path = std::string(testing::TempDir()) + "/sf.gguf";
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

  ggml_init_params ip = {32 * 1024 * 1024, nullptr, false};
  ggml_context* ctx = ggml_init(ip);
  int seed = 3;
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

// 物理流式执行的正确性锚点:流式生成必须与常驻(GreedyGenerateCached)逐 token 一致。
TEST(StreamingForwardTest, StreamingMatchesResident) {
  const std::string path = WriteTinyOlmoe();
  auto sm = StreamingModel::Load(path);
  auto full = OlmoeModel::Load(path);
  ASSERT_NE(sm, nullptr);
  ASSERT_NE(full, nullptr);

  std::vector<int32_t> prompt = {3, 7, 2};
  for (bool norm_topk : {false, true}) {
    // capacity ≥ n_expert_used(=2);取 3 顺带走一次淘汰路径。
    std::vector<int32_t> streamed =
        StreamingGenerate(*sm, prompt, /*n_predict=*/6, norm_topk, /*capacity=*/3);
    std::vector<int32_t> resident =
        GreedyGenerateCached(*full, prompt, 6, norm_topk);
    EXPECT_EQ(streamed, resident) << "norm_topk=" << norm_topk;
  }

  std::remove(path.c_str());
}

TEST(StreamingForwardTest, EmptyOrZeroPredict) {
  const std::string path = WriteTinyOlmoe();
  auto sm = StreamingModel::Load(path);
  ASSERT_NE(sm, nullptr);
  EXPECT_TRUE(StreamingGenerate(*sm, {}, 4, false, 3).empty());
  EXPECT_TRUE(StreamingGenerate(*sm, {1}, 0, false, 3).empty());
  std::remove(path.c_str());
}

}  // namespace
}  // namespace nuthatch
