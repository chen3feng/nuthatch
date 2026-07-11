#include "src/model/streaming_model.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "gtest/gtest.h"
#include "src/model/expert_reader.h"
#include "src/model/olmoe_model.h"

namespace nuthatch {
namespace {

// 写一个 tiny OLMoE GGUF(含专家 3D 张量与非专家张量)。
std::string WriteTinyOlmoe() {
  const std::string path = std::string(testing::TempDir()) + "/sm.gguf";
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
  int seed = 11;
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

TEST(StreamingModelTest, ResidentNonExpertMatchesFullExpertsStreamed) {
  const std::string path = WriteTinyOlmoe();
  auto full = OlmoeModel::Load(path);
  auto sm = StreamingModel::Load(path);
  ASSERT_NE(full, nullptr);
  ASSERT_NE(sm, nullptr);

  EXPECT_EQ(sm->config().n_layers, full->config().n_layers);
  EXPECT_EQ(sm->config().n_embd, full->config().n_embd);
  EXPECT_EQ(sm->config().n_vocab, full->config().n_vocab);
  EXPECT_EQ(sm->config().n_expert, full->config().n_expert);

  // 非专家张量:常驻且逐字节等同全量加载。
  for (const char* name :
       {"token_embd.weight", "output.weight", "output_norm.weight",
        "blk.0.attn_q.weight", "blk.1.ffn_gate_inp.weight",
        "blk.1.ffn_norm.weight"}) {
    ggml_tensor* a = sm->tensor(name);
    ggml_tensor* b = full->tensor(name);
    ASSERT_NE(a, nullptr) << name;
    ASSERT_NE(b, nullptr) << name;
    ASSERT_EQ(ggml_nbytes(a), ggml_nbytes(b));
    EXPECT_EQ(std::memcmp(a->data, b->data, ggml_nbytes(a)), 0) << name;
  }

  // 专家张量:StreamingModel 不常驻,但可经 reader 流式读到、且字节一致。
  for (const char* en :
       {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
        "blk.1.ffn_down_exps.weight"}) {
    EXPECT_EQ(sm->tensor(en), nullptr) << en;
    const size_t eb = sm->reader()->ExpertBytes(en);
    ASSERT_GT(eb, 0u);
    std::vector<char> buf(eb);
    ASSERT_TRUE(sm->reader()->ReadExpert(en, 2, buf.data()));
    ggml_tensor* fe = full->tensor(en);
    ASSERT_NE(fe, nullptr);
    EXPECT_EQ(std::memcmp(buf.data(),
                          static_cast<const char*>(fe->data) + 2 * fe->nb[2], eb),
              0)
        << en;
  }

  std::remove(path.c_str());
}

TEST(StreamingModelTest, LoadFailsOnMissingFile) {
  EXPECT_EQ(StreamingModel::Load("/no/such/model.gguf"), nullptr);
}

}  // namespace
}  // namespace nuthatch
