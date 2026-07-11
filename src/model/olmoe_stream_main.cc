// 物理流式推理 demo:专家不常驻,推理时按需从盘装进有界槽缓存(容量/层)。
// 输出文本 + 峰值常驻内存(证明真省 RAM,常驻路径需 ~4GB)。速度不是重点——
// 逐 token 多趟小图 + miss 时磁盘读,慢于常驻;价值是"真在受限内存里跑起来"。
//   olmoe_stream <model.gguf> <n_predict> <capacity/层> <prompt...>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/model/streaming_forward.h"
#include "src/model/streaming_model.h"
#include "src/tokenizer/tokenizer.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s model.gguf n_predict capacity prompt...\n",
                 argv[0]);
    return 1;
  }
  auto model = nuthatch::StreamingModel::Load(argv[1]);
  auto tok = nuthatch::Tokenizer::Load(argv[1]);
  if (model == nullptr || tok == nullptr) {
    std::fprintf(stderr, "load failed: %s\n", argv[1]);
    return 1;
  }
  const int n_predict = std::atoi(argv[2]);
  const int capacity = std::atoi(argv[3]);

  std::string prompt;
  for (int i = 4; i < argc; ++i) {
    if (i > 4) prompt += ' ';
    prompt += argv[i];
  }
  std::vector<int32_t> ids = tok->Encode(prompt);

  const char* nt = std::getenv("NUTHATCH_NORM_TOPK");
  const bool norm_topk = (nt != nullptr && nt[0] == '1');

  std::vector<int32_t> gen =
      nuthatch::StreamingGenerate(*model, ids, n_predict, norm_topk, capacity);

  std::vector<int32_t> full = ids;
  full.insert(full.end(), gen.begin(), gen.end());
  std::printf("text: %s\n", tok->Decode(full).c_str());

#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc = {};
  GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
  const double mb = pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
#elif defined(__APPLE__)
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  const double mb = ru.ru_maxrss / (1024.0 * 1024.0);  // macOS:字节
#else
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  const double mb = ru.ru_maxrss / 1024.0;  // Linux:KB
#endif
  std::printf("peak RSS: %.0f MB (capacity=%d 专家/层,非专家常驻)\n", mb,
              capacity);
  return 0;
}
