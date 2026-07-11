#include "src/model/olmoe_model.h"

#include <cstdio>
#include <string>

#include "ggml.h"
#include "gguf.h"
#include "gtest/gtest.h"

namespace nuthatch {
namespace {

// 造一个【结构完整、维度极小】的 OLMoE GGUF(2 层),用于离线/CI 测加载器
// (CI 上没有 4GB 真模型)。张量名/形状与真 OLMoE 一致,只是维度缩小。
struct TinyDims {
  int n_embd = 8, n_head = 2, n_ff = 4, n_expert = 4, n_used = 2, n_vocab = 16,
      n_layers = 2;
};

std::string WriteTinyOlmoe(const TinyDims& d) {
  const std::string path = std::string(testing::TempDir()) + "/tiny_olmoe.gguf";
  gguf_context* w = gguf_init_empty();

  gguf_set_val_str(w, "general.architecture", "olmoe");
  gguf_set_val_u32(w, "olmoe.block_count", d.n_layers);
  gguf_set_val_u32(w, "olmoe.embedding_length", d.n_embd);
  gguf_set_val_u32(w, "olmoe.attention.head_count", d.n_head);
  gguf_set_val_u32(w, "olmoe.attention.head_count_kv", d.n_head);
  gguf_set_val_u32(w, "olmoe.feed_forward_length", d.n_ff);
  gguf_set_val_u32(w, "olmoe.expert_count", d.n_expert);
  gguf_set_val_u32(w, "olmoe.expert_used_count", d.n_used);
  gguf_set_val_u32(w, "olmoe.context_length", 32);
  gguf_set_val_f32(w, "olmoe.rope.freq_base", 10000.0f);
  gguf_set_val_f32(w, "olmoe.attention.layer_norm_rms_epsilon", 1e-5f);

  ggml_init_params ip = {/*mem_size=*/32 * 1024 * 1024,
                         /*mem_buffer=*/nullptr, /*no_alloc=*/false};
  ggml_context* ctx = ggml_init(ip);

  auto add2 = [&](const char* name, int ne0, int ne1) {
    ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name);
    gguf_add_tensor(w, t);
  };
  auto add1 = [&](const char* name, int ne0) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(t, name);
    gguf_add_tensor(w, t);
  };
  auto add3 = [&](const std::string& name, int ne0, int ne1, int ne2) {
    ggml_tensor* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(t, name.c_str());
    gguf_add_tensor(w, t);
  };

  add2("token_embd.weight", d.n_embd, d.n_vocab);
  add2("output.weight", d.n_embd, d.n_vocab);
  add1("output_norm.weight", d.n_embd);
  for (int l = 0; l < d.n_layers; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    add1((b + "attn_norm.weight").c_str(), d.n_embd);
    add2((b + "attn_q.weight").c_str(), d.n_embd, d.n_embd);
    add2((b + "attn_k.weight").c_str(), d.n_embd, d.n_embd);
    add2((b + "attn_v.weight").c_str(), d.n_embd, d.n_embd);
    add2((b + "attn_output.weight").c_str(), d.n_embd, d.n_embd);
    add1((b + "attn_q_norm.weight").c_str(), d.n_embd);
    add1((b + "attn_k_norm.weight").c_str(), d.n_embd);
    add1((b + "ffn_norm.weight").c_str(), d.n_embd);
    add2((b + "ffn_gate_inp.weight").c_str(), d.n_embd, d.n_expert);  // router
    add3(b + "ffn_gate_exps.weight", d.n_embd, d.n_ff, d.n_expert);
    add3(b + "ffn_up_exps.weight", d.n_embd, d.n_ff, d.n_expert);
    add3(b + "ffn_down_exps.weight", d.n_ff, d.n_embd, d.n_expert);
  }

  gguf_write_to_file(w, path.c_str(), /*only_meta=*/false);
  ggml_free(ctx);
  gguf_free(w);
  return path;
}

TEST(OlmoeModelTest, LoadsConfigAndWeights) {
  TinyDims d;
  const std::string path = WriteTinyOlmoe(d);

  auto m = OlmoeModel::Load(path);
  ASSERT_NE(m, nullptr);

  const OlmoeConfig& c = m->config();
  EXPECT_EQ(c.n_layers, 2u);
  EXPECT_EQ(c.n_embd, 8u);
  EXPECT_EQ(c.n_head, 2u);
  EXPECT_EQ(c.n_head_kv, 2u);
  EXPECT_EQ(c.head_dim(), 4u);
  EXPECT_EQ(c.n_ff, 4u);
  EXPECT_EQ(c.n_expert, 4u);
  EXPECT_EQ(c.n_expert_used, 2u);
  EXPECT_EQ(c.n_vocab, 16u);  // 从 output.weight ne[1] 推
  EXPECT_FLOAT_EQ(c.rope_freq_base, 10000.0f);
  EXPECT_FLOAT_EQ(c.rms_eps, 1e-5f);

  // 关键权重存在且形状正确。
  ASSERT_NE(m->tensor("token_embd.weight"), nullptr);
  ASSERT_NE(m->tensor("output_norm.weight"), nullptr);

  ggml_tensor* q = m->layer_tensor(0, "attn_q.weight");
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(q->ne[0], 8);
  EXPECT_EQ(q->ne[1], 8);

  ggml_tensor* gate = m->layer_tensor(1, "ffn_gate_exps.weight");
  ASSERT_NE(gate, nullptr);
  EXPECT_EQ(gate->ne[0], 8);  // n_embd
  EXPECT_EQ(gate->ne[1], 4);  // n_ff
  EXPECT_EQ(gate->ne[2], 4);  // n_expert

  ggml_tensor* router = m->layer_tensor(0, "ffn_gate_inp.weight");
  ASSERT_NE(router, nullptr);
  EXPECT_EQ(router->ne[1], 4);  // n_expert

  // 不存在的张量返回 nullptr。
  EXPECT_EQ(m->tensor("nope.weight"), nullptr);
  EXPECT_EQ(m->layer_tensor(9, "attn_q.weight"), nullptr);

  std::remove(path.c_str());
}

TEST(OlmoeModelTest, ReturnsNullOnMissingFile) {
  EXPECT_EQ(OlmoeModel::Load("/no/such/olmoe.gguf"), nullptr);
}

}  // namespace
}  // namespace nuthatch
