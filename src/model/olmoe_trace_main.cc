// 护城河证据工具:从【真实 OLMoE 推理】导出专家路由 trace,用 M2 的三种缓存
// 策略(learned-pin / per-layer LRU / OS 全局页缓存)在同一 trace、同一总预算下
// 重放,对比命中率(miss = 需从磁盘读专家)。如实报数——OLMoE 训练做了专家负载
// 均衡,真实使用未必像合成 trace 那样偏斜,learned 优势多大由数据说话。
//
//   olmoe_trace <model.gguf> <n_predict> <budget/层> <prompt...>
//   env NUTHATCH_TRACE_OUT=path 可另存 trace 二进制。
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/cache/cache_policy.h"
#include "src/cache/learned_pin_policy.h"
#include "src/cache/lru_policy.h"
#include "src/cache/os_page_cache_policy.h"
#include "src/model/generate.h"
#include "src/model/olmoe_model.h"
#include "src/tokenizer/tokenizer.h"
#include "src/trace/routing_trace.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s model.gguf n_predict budget_per_layer prompt...\n",
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
  const uint32_t budget = static_cast<uint32_t>(std::atoi(argv[3]));

  std::string prompt;
  for (int i = 4; i < argc; ++i) {
    if (i > 4) prompt += ' ';
    prompt += argv[i];
  }
  std::vector<int32_t> ids = tok->Encode(prompt);

  const char* nt = std::getenv("NUTHATCH_NORM_TOPK");
  const bool norm_topk = (nt != nullptr && nt[0] == '1');

  // 真实推理 + 导出路由 trace。
  nuthatch::RoutingTrace tr;
  std::vector<int32_t> gen =
      nuthatch::GreedyGenerateCachedTrace(*model, ids, n_predict, norm_topk, &tr);

  std::vector<int32_t> full = ids;
  full.insert(full.end(), gen.begin(), gen.end());
  std::printf("text: %s\n", tok->Decode(full).c_str());
  std::printf("trace: %zu records over %u layers, %u experts/layer, budget=%u/layer\n",
              tr.records.size(), tr.n_layers, tr.n_expert, budget);

  if (const char* out = std::getenv("NUTHATCH_TRACE_OUT")) {
    std::printf("trace written: %s (%s)\n", out,
                nuthatch::WriteRoutingTrace(tr, out) ? "ok" : "FAILED");
  }

  // 同一总预算(budget/层 × 层数)下三策略重放。learned:多数槽 pin 历史最热,
  // 余下 LRU;lru:每层 budget 槽;os:单个全局 LRU,容量 = budget×层数。
  const uint32_t pin = budget > 1 ? (budget * 3 + 3) / 4 : 1;  // ≈75% 预留 pin
  const uint32_t lru_slots = budget > pin ? budget - pin : 0;
  const nuthatch::UsageHistogram usage = nuthatch::BuildUsage(tr);

  nuthatch::LearnedPinPolicy learned(tr.n_layers, pin, lru_slots, usage);
  nuthatch::LruPolicy lru(tr.n_layers, budget);
  nuthatch::OsPageCachePolicy os(static_cast<uint64_t>(budget) * tr.n_layers);

  const nuthatch::ReplayStats sl = nuthatch::Replay(tr, &learned);
  const nuthatch::ReplayStats sr = nuthatch::Replay(tr, &lru);
  const nuthatch::ReplayStats so = nuthatch::Replay(tr, &os);

  std::printf("\n           hit-rate   hits/accesses   (miss = disk read)\n");
  auto row = [](const char* name, const nuthatch::ReplayStats& s) {
    std::printf("%-10s  %6.1f%%   %llu/%llu\n", name, 100.0 * s.hit_rate(),
                static_cast<unsigned long long>(s.hits),
                static_cast<unsigned long long>(s.accesses));
  };
  row("learned", sl);
  row("lru", sr);
  row("os-page", so);
  std::printf("\nlearned vs lru:  %+.1f pp   vs os: %+.1f pp\n",
              100.0 * (sl.hit_rate() - sr.hit_rate()),
              100.0 * (sl.hit_rate() - so.hit_rate()));
  return 0;
}
