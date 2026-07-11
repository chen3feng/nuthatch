// 本地推理工具(全自包含):文本 prompt → 分词 → 贪心生成 → 解码回文本。
//   olmoe_generate <model.gguf> <n_predict> <prompt text ...>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/model/generate.h"
#include "src/model/olmoe_model.h"
#include "src/tokenizer/tokenizer.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s model.gguf n_predict prompt text ...\n",
                 argv[0]);
    return 1;
  }
  auto model = nuthatch::OlmoeModel::Load(argv[1]);
  auto tok = nuthatch::Tokenizer::Load(argv[1]);
  if (model == nullptr || tok == nullptr) {
    std::fprintf(stderr, "load failed: %s\n", argv[1]);
    return 1;
  }
  const int n_predict = std::atoi(argv[2]);

  // argv[3..] 拼成 prompt 文本,分词成 ids。
  std::string prompt;
  for (int i = 3; i < argc; ++i) {
    if (i > 3) prompt += ' ';
    prompt += argv[i];
  }
  std::vector<int32_t> ids = tok->Encode(prompt);

  // OLMoE norm_topk_prob=false(env NUTHATCH_NORM_TOPK=1 可临时开)。
  const char* nt = std::getenv("NUTHATCH_NORM_TOPK");
  const bool norm_topk = (nt != nullptr && nt[0] == '1');

  std::printf("prompt ids:");
  for (int32_t t : ids) std::printf(" %d", t);
  std::printf("\n");

  std::vector<int32_t> gen =
      nuthatch::GreedyGenerate(*model, ids, n_predict, norm_topk);

  std::printf("generated ids:");
  for (int32_t t : gen) std::printf(" %d", t);
  std::printf("\n");

  std::vector<int32_t> full = ids;
  full.insert(full.end(), gen.begin(), gen.end());
  std::printf("text: %s\n", tok->Decode(full).c_str());
  return 0;
}
