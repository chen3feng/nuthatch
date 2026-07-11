// 本地调试工具:加载一个 OLMoE GGUF,对给定 token id 序列跑前向,
// 打印最后位置的 top-5 next-token logits。用来在【真模型】上确认前向产出
// 合理(峰值分布而非乱码)。CI 不跑(无真模型),仅需编译通过。
//
//   olmoe_forward <model.gguf> <tok0> <tok1> ...
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "src/model/forward.h"
#include "src/model/olmoe_model.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.gguf tok0 [tok1 ...]\n", argv[0]);
    return 1;
  }
  auto model = nuthatch::OlmoeModel::Load(argv[1]);
  if (model == nullptr) {
    std::fprintf(stderr, "load failed: %s\n", argv[1]);
    return 1;
  }
  const nuthatch::OlmoeConfig& cfg = model->config();
  std::printf("loaded: layers=%u n_embd=%u n_expert=%u/%u vocab=%u\n",
              cfg.n_layers, cfg.n_embd, cfg.n_expert_used, cfg.n_expert,
              cfg.n_vocab);

  std::vector<int32_t> toks;
  for (int i = 2; i < argc; ++i) toks.push_back(std::atoi(argv[i]));
  const int T = static_cast<int>(toks.size());

  ggml_init_params ip = {static_cast<size_t>(2) * 1024 * 1024 * 1024, nullptr,
                         false};
  ggml_context* ctx = ggml_init(ip);
  ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
  ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
  ggml_tensor* logits =
      nuthatch::BuildForward(ctx, *model, ids, pos, /*norm_topk=*/true);
  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, logits);

  for (int i = 0; i < T; ++i) {
    static_cast<int32_t*>(ids->data)[i] = toks[i];
    static_cast<int32_t*>(pos->data)[i] = i;
  }
  ggml_graph_compute_with_ctx(ctx, gf, 4);

  const int V = static_cast<int>(cfg.n_vocab);
  const float* last = static_cast<float*>(logits->data) + (T - 1) * V;
  std::vector<int> idx(V);
  for (int i = 0; i < V; ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                    [&](int a, int b) { return last[a] > last[b]; });
  std::printf("top-5 next-token logits (after pos %d):\n", T - 1);
  for (int i = 0; i < 5; ++i) {
    std::printf("  tok %-6d logit %.4f\n", idx[i], last[idx[i]]);
  }

  ggml_free(ctx);
  return 0;
}
