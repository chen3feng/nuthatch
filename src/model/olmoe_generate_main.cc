// 本地推理工具:加载真 OLMoE,对给定 prompt token id 序列贪心生成 n 个 token,
// 打印生成的 token id。配合 llama-tokenize/llama-cli 做 token-exact 对拍。
//
//   olmoe_generate <model.gguf> <n_predict> <tok0> [tok1 ...]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/model/generate.h"
#include "src/model/olmoe_model.h"
#include "src/tokenizer/tokenizer.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s model.gguf n_predict tok0 [tok1 ...]\n",
                 argv[0]);
    return 1;
  }
  auto model = nuthatch::OlmoeModel::Load(argv[1]);
  if (model == nullptr) {
    std::fprintf(stderr, "load failed: %s\n", argv[1]);
    return 1;
  }
  const int n_predict = std::atoi(argv[2]);
  std::vector<int32_t> ids;
  for (int i = 3; i < argc; ++i) ids.push_back(std::atoi(argv[i]));

  // OLMoE 不归一化 top-k 权重(norm_topk_prob=false)——对拍 llama.cpp 定住:
  // norm_topk=true 时首 token 分歧(called vs Paris),false 时 token-exact。
  // env NUTHATCH_NORM_TOPK=1 可临时开启(调试其他 arch 用)。
  const char* nt = std::getenv("NUTHATCH_NORM_TOPK");
  const bool norm_topk = (nt != nullptr && nt[0] == '1');

  std::vector<int32_t> gen =
      nuthatch::GreedyGenerate(*model, ids, n_predict, norm_topk);

  std::printf("generated ids:");
  for (int32_t t : gen) std::printf(" %d", t);
  std::printf("\n");

  // 若模型带词表,把 prompt + 生成解码成文本打印(自包含输出)。
  if (auto tok = nuthatch::Tokenizer::Load(argv[1])) {
    std::vector<int32_t> full = ids;
    full.insert(full.end(), gen.begin(), gen.end());
    std::printf("text: %s\n", tok->Decode(full).c_str());
  }
  return 0;
}
