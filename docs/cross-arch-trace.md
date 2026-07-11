# 跨架构验证:从别的 MoE 导出路由 trace

nuthatch 引擎只实现了 OLMoE 架构,但护城河(learned 专家缓存)的结论**不该**是
OLMoE 专属。要在**别的 MoE 架构**上验证,只需拿到它的路由 trace(每 token 每层
选中的专家),再喂 `trace_replay --text`——不需要把那个模型跑进 nuthatch 引擎。

路由 trace 从 **llama.cpp** 导出(它支持几十种 MoE):给它的 `eval-callback` 示例
打个小补丁,eval 时把名为 `ffn_moe_topk` 的张量(每层 top-k 选中的专家 id)读出来。

## 步骤

1. 拿一个小 MoE 的 GGUF。例:`ollama pull granite3-moe:1b`(IBM Granite 3.0 MoE,
   ~821MB,24 层 / 32 专家 / 选 8),GGUF 在 `~/.ollama/models/blobs/sha256-<model>`。

2. 给 llama.cpp 的 `examples/eval-callback/eval-callback.cpp` 打补丁:
   - 加一个只 dump `ffn_moe_topk` 的回调(`ask` 阶段只对该张量返回 true;`ask=false`
     时 `ggml_backend_tensor_get` 读全量 `[n_expert_used, n_tokens]` I32,按层号
     `ffn_moe_topk-<il>` 解析 layer,每 token 一列输出一行 `"<layer> e0 e1 ..."`)。
   - `run()` 里 prefill 后加**贪心生成**循环(拿 decode 阶段路由)。
   - `main()` 里 env `NUTHATCH_MOE_TRACE` 打开输出文件、把 `cb_eval` 指向该回调。
   - (完整补丁见下方"回调核心"。)

3. 建并运行:
   ```bash
   cmake -B build-cpu -DGGML_METAL=OFF -DLLAMA_CURL=OFF && \
     cmake --build build-cpu --target llama-eval-callback -j
   NUTHATCH_MOE_TRACE=granite.trace.txt \
     build-cpu/bin/llama-eval-callback -m <granite.gguf> -p "Once upon a time" -n 64
   ```

4. 喂 nuthatch 的 model-free 工具:
   ```bash
   trace_replay --text granite.trace.txt 4 8 12 16      # 32 专家:12.5%~50%
   ```

本仓已附一份导出好的 trace:[`data/granite3moe.trace.txt`](../data/granite3moe.trace.txt)
(1629 records / 24 层 / 32 专家),可直接 `trace_replay --text` 复现下方结果。

## 结果(见 `bench-log.md`)

granite3-moe 上,learned-pin 稳定赢 LRU/OS **+8.8 ~ +17.7 pp**,且**预算越紧优势
越大**(12.5% 预算时 OS 塌到 0%、LRU 4.7%、learned 仍 22.4%)——与 OLMoE 同规律。
**护城河跨架构成立。**

## 回调核心(补丁要点)

```cpp
static FILE* g_moe_trace = nullptr;  // env NUTHATCH_MOE_TRACE 打开

static bool moe_trace_cb(ggml_tensor* t, bool ask, void*) {
  const bool is_topk = std::strncmp(t->name, "ffn_moe_topk", 12) == 0;
  if (ask) return is_topk;                 // 只索取这些张量
  if (!is_topk || !g_moe_trace) return true;
  int il = 0; const char* d = std::strrchr(t->name, '-'); if (d) il = atoi(d+1);
  const int64_t n_used = t->ne[0], n_tok = t->ne[1];
  std::vector<int32_t> buf(n_used * n_tok);
  ggml_backend_tensor_get(t, buf.data(), 0, buf.size()*sizeof(int32_t));
  for (int64_t k = 0; k < n_tok; ++k) {    // 每 token 一行
    std::fprintf(g_moe_trace, "%d", il);
    for (int64_t e = 0; e < n_used; ++e) std::fprintf(g_moe_trace, " %d", buf[k*n_used+e]);
    std::fprintf(g_moe_trace, "\n");
  }
  return true;
}
```

注:cache 重放只看 `(layer, expert)` 的**时序**,不需精确 token 序号,故按回调
顺序逐行输出即可(每层的相对顺序天然保持)。
