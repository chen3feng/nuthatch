# nuthatch — 研究日志

> **nuthatch**(䴓):把种子塞进树皮缝里囤着、记得位置的鸟;头朝下爬树。
> *tagline: caches what matters, streams the rest.*
> 代号取自"囤食 + 记位置"= usage-learned 专家缓存;延续 colibrì 的鸟血统。

> 目标:实现一个**支持 GGUF、跨 Mac/Linux/Windows、C++(blade-build)** 的 MoE 流式推理引擎。
> 核心思想沿用 [colibrì](https://github.com/JustVugg/colibri)——稠密常驻、路由专家留盘按需流式,但**原生读 GGUF、原生 k-quant、多后端(Metal/CUDA/Vulkan via ggml)**。
> 双目标:**学手艺**(把 GGUF/后端/图/流式全趟一遍)+ **涨 credit**(回答 colibrì 没测过的问题)。
> 护城河 = "usage-learned 逐专家缓存 + 预取"这个**算法**,不是引擎本身(引擎是 ggml/llama.cpp 已解决的 commodity)。

---

## 测试环境

| 项 | 值 |
|---|---|
| 机器 | Apple M4 Pro · 48 GB 统一内存 · macOS |
| GPU | Metal · MTLGPUFamilyApple9 · `has tensor = false`(pre-M5,走标准 Metal,非新 tensor 路径) |
| 引擎 | llama.cpp(brew,ggml 0.16.0) · iobench(colibrì 自带,编在 `/tmp/iobench`) |
| 主测模型 | OLMoE-1B-7B-0924-Instruct **Q4_K_M** · 3.92 GiB · 6.92B(激活 1.7B)· 16 层 · 64 专家 |
| 模型缓存路径 | `~/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924-Instruct-GGUF/`(勿删,流量有限) |

---

## 实测数据(2026-07-11)

### ① 磁盘带宽(iobench,19MB 随机块 = 专家读法)

| 目标 | 模式 | 带宽 | 备注 |
|---|---|---|---|
| **M4 Pro 内置 SSD** | O_DIRECT ×8,`sudo purge` 后 | **6.51 GB/s** | 真盘数,已越过 ~5 GB/s 拐点 |
| 暖命中(污染测得) | buffered ×8 | ~46 GB/s | = 统一内存带宽,即"命中后"的读速 |
| 〃 | O_DIRECT ×1(未清缓存) | ~14 GB/s | 单线程内存带宽,非真盘 |

> ⚠️ macOS 的 `F_NOCACHE` **不驱逐已缓存页**;文件 <RAM 时必须 `sudo purge` 才能量到真盘。

存储档位对照(colibrì benchmark 表 + 本机实测):

| 存储 | 带宽 | 档 |
|---|---|---|
| 外置 SSD(用户另一块) | ~2 GB/s | 低-中,死磁盘受限 |
| Optane 905p | 3.3 GB/s | 中 |
| **M4 Pro 内置(本机)** | **6.51 GB/s** | 快,过拐点 |
| Samsung 9100 PRO(PCIe5) | 8.8 GB/s | 快 |
| M5 Max 内置 | 14 GB/s | 最快 |

### ② CPU vs Metal(OLMoE 全常驻,llama-bench)

| 阶段 | CPU(`-ngl 0`) | Metal(`-ngl 99`) | 加速 |
|---|---|---|---|
| prefill pp512 | ~212 t/s | ~2283 t/s | **10.7×** |
| decode tg128 | ~101 t/s | ~224 t/s | **2.2×** |

> prefill 计算密集 → Metal 狂赢;decode 是 GEMV、内存带宽受限 → Metal 只赢 2.2×。
> 这是"命中率 = 100%"的**右端锚点**。

### ③ `-ngl` 扫描(GPU 层数 vs 速度)——**decode 低谷**

`-ngl` = `--n-gpu-layers` = 放几层到 GPU(0=全 CPU,99=全 GPU;OLMoE 16 层)。

| ngl | pp512 | tg128 |
|---|---|---|
| 0 | 212 | **101** |
| 4 | 224 | **75** ⚠️ 低谷:比纯 CPU 还慢 |
| 8 | 320 | 96 |
| 12 | 514 | 101 |
| 16 | 1331 | 183 |
| 99 | 2283 | 224 |

---

## 推导:GLM-5.2 @ 6.51 GB/s

成本模型:decode 冷读 ~11 GB/token。Amdahl:`总加速 = 1/(磁盘占比 + 计算占比/S)`,S≈2.2(偏乐观,colibrì 的 NEON IDOT CPU 基线更高,真实可能 ~1.5–2)。

| 场景 | 磁盘占比 | tok/s 估 | MPS 加速估 |
|---|---|---|---|
| 冷读 | ~40% | `11÷6.51`≈**0.6** | ~1.4× |
| 暖 + pin + MTP | 塌缩(读→46 GB/s 内存带宽) | ~1–1.5 | **→2×** |

> **本机是唯一能把 "MPS 1.4×→2×" 整条曲线测出来的配置**:外置盘太慢(压平在 1.2×),本机内置盘恰在拐点。

---

## 关键结论 / 已学到的教训

1. **MPS 只加速"计算",碰不到"磁盘"** → 对磁盘受限的 GLM-5.2,MPS 是 ~1.2–2×,不是大数字;且**只有命中率上去后才兑现**。正确顺序:先做 pin/学习缓存 → 读变内存带宽级 → 再上 MPS 收割。
2. **decode 低谷(ngl=4 < ngl=0)**:部分层上 GPU 比纯 CPU 还慢。设计警告:流式引擎的 CPU/GPU 边界**别 per-layer ping-pong**;要么整层归一设备,要么用 **MTP 把 decode 批量化**摊薄交接开销。
3. **统一内存消掉的是"拷贝",不是"同步"**:CPU/GPU 仍是两个处理器,交接要 command 派发 + 完成 fence + (decode 时)GPU 空转。这些跟数据搬不搬无关。→ 相对 CUDA 的真优势是"无拷贝",但交接成本还在。
4. **6.51 GB/s 过拐点**:本机 GLM-5.2 从"磁盘受限"迈入"磁盘/计算平衡",是第一个 MPS 值得测的机器。
5. **格式**:项目应"**读 GGUF**"而非"转 GGUF"。转换器是 colibrì 因无 GGUF 生态才被迫写的包袱,拥抱 GGUF 正为甩掉它。

---

## 相关工作 / prior art

- **MC-SMoE**(ICLR'24 Spotlight,"Merge, Then Compress")— [github](https://github.com/UNITES-Lab/MC-SMoE)。用路由统计**合并冗余专家** + 低秩/稀疏压缩,号称省 80% 内存、几乎无损(但**只在 SST-2/MRPC/SQuAD 等简单 benchmark 上**,硬生成任务存疑)。Python/PyTorch,针对 **Switch Transformer + Mixtral-8x7B**。
  - **与本项目的关系**:**不同轴、互补**。它"把模型改小"(算法/模型手术、有损),我"绕开大模型"(系统/运行时、无损)。可作**上游预处理**:256 专家合并到 64 → 流式问题缩小 4×、命中率更高。
  - **潜在 credit 点(交叉真空)**:**"专家合并会不会改变学习缓存的价值?"** 合并去冗余后,剩下的专家更独特——缓存是更管用(专家少)还是更没用(每个都必需)?没人测过。等引擎能跑后,拿合并前/后的同一模型跑命中率曲线即可。先在 Mixtral 上做(它原生支持)。

## 架构决定(暂定)

- **建在 ggml(低层)之上,不建在 llama.cpp(高层)之上**:llama.cpp 假设权重全 mmap 常驻,做逐专家流式是跟它核心假设对着干。ggml 给张量 + k-quant + `ggml_backend`(多后端免费),per-arch 图和流式自己写。
- 依赖获取:**vcpkg** `ggml[metal]`(Mac)/ `ggml[cuda,vulkan]`(Linux/Win),v0.11.1,MIT 许可。
- **最小外科切入点**:实现一个自定义 ggml **backend buffer**,让专家张量按需 fault-in,其余尽量复用。
- **GGUF 专家是融合 3D 张量**(`blk.N.ffn_*_exps.weight`)+ `ggml_mul_mat_id`;流式要么切片单专家(丢融合核)、要么做懂 super-block 对齐的惰性 buffer。

---

## 下一步 / 待办

- [ ] 免下载:把 ①②③ 画成第一张图(prefill/decode 两条,标 decode 低谷)。
- [ ] 用 llama.cpp `--n-cpu-moe` / `-ot exps=CPU` 量 OLMoE 的"专家在 CPU"基线(对照组)。
- [ ] 设计 routing-trace 格式 + 重放器,离线对比 `OS页缓存 / LRU / LRU+learned-pin` 命中率曲线(几乎不占盘)。
- [ ] (需快盘或命中率高)测 "MPS 加速 × 命中率" 曲线——本机内置盘可做。
- [ ] Windows I/O 路径(`ReadFile` + `OVERLAPPED` + `FILE_FLAG_NO_BUFFERING`)——colibrì 完全没做。
- [ ] 严谨测 per-row int4 vs Q4_K_M 的精度差(colibrì 明确求助、没做的事)。
- [ ] (交叉研究)在 Mixtral 上:MC-SMoE 合并前/后,流式学习缓存命中率曲线怎么变?

### 2026-07-11 — KV cache:~2× decode 提速,且大 T 更稳(P18)

- 机器/存储:M4 Pro,48GB,内置 SSD;模型:OLMoE-1B-7B Instruct Q4_K_M(常驻)。
- 命令:`olmoe_generate <model> <N> The capital of France is`(cached 默认;`NUTHATCH_NO_KV_CACHE=1` 切每步重算)。
- 结果(prompt "The capital of France is",贪心):
  - **生成 id 完全一致**:cached 与 no-cache 逐 token 相同(→ `Paris.`),token-exact。
  - **速度**(N=12):cached **22.5 tok/s** vs no-cache **11.5 tok/s** ≈ **×2**。
  - **稳健性**:N=24 时 no-cache 在 `BuildMoe`→`ggml_add` 崩(每步整段前向的图 ctx 在大 T 下不够);cached 每步只前向 1 token,稳跑 **38 tok/s**。
- 结论:KV cache 让 decode 从 O(T)/步 降到 O(1)/步(读缓存 K/V),既更快又避开了重算路径的大-T 脆弱性。正确性锚点 `CachedMatchesUncached` 单测护住。
- 待办:每 decode 步仍重建 ggml_context(128MB init)——后续可复用预分配 compute buffer 进一步提速。

### 2026-07-11 — ★ learned-pin 在【真实 OLMoE 推理】上赢 LRU/OS 6–9pp(P21)

- 机器/模型:M4 Pro;OLMoE-1B-7B Instruct Q4_K_M。16 层、64 专家/层、选 8。
- 方法:`olmoe_trace` 真实推理(prompt "Once upon a time" + 64 decode token)导出
  路由 trace(1072 records = 68 token × 16 层,8576 次专家访问),用 M2 三策略在
  **同一 trace、同一总预算**(budget/层 × 16 层)下重放比命中率。miss = 需读盘。
- 结果(命中率):

  | 预算/层 (占 64) | learned-pin | per-layer LRU | OS 全局页缓存 | learned 优势 |
  |---|---|---|---|---|
  | 8  (12.5%) | **32.7%** | 23.7% | 23.7% | **+9.1 pp** |
  | 16 (25%)   | **54.3%** | 48.0% | 47.1% | +6.4 / +7.2 pp |
  | 24 (37.5%) | **70.9%** | 62.1% | 61.8% | +8.8 / +9.1 pp |

- 结论:**M2 的模拟结论在真实推理路径上复现**——即便 OLMoE 训练做了专家负载
  均衡,真实使用仍有足够偏斜,把历史最热专家 pin 常驻稳定甩开 LRU/OS 6–9pp。
  这是"用量学习的专家流式缓存"差异化的**真实证据**(非合成 trace)。
- 复现:`olmoe_trace <model> 64 <budget/层> Once upon a time`。
- 待办(可选工程):物理显存受限执行(真在槽上 mul_mat_id,省内存);更多 prompt/
  更长序列的稳健性;pin/lru 配比与 budget 的完整曲线。

### 2026-07-11 — ★ budget 曲线 + 跨领域稳健:预算越紧 learned 优势越大(P22)

- 方法:`olmoe_trace` 对 3 个不同领域 prompt 各生成 64-decode trace(story/fact/code),
  `trace_replay`(model-free)扫每层预算 4→32、比三策略命中率。
- learned − LRU 优势(百分点):

  | 预算/层 (占 64) | story | fact | code |
  |---|---|---|---|
  | 4  (6%)  | +17.5 | +15.5 | **+24.2** |
  | 8  (12%) | +9.1  | +9.4  | +15.7 |
  | 16 (25%) | +6.4  | +8.4  | +10.6 |
  | 32 (50%) | +8.1  | +11.4 | +7.9  |

- ★ 关键发现:**预算越紧,learned 优势越大**。budget=4 时 OS 全局页缓存 **0% 命中**
  (64 槽全局池彻底 thrash)、per-layer LRU 也仅 3–5%,而 learned pin 住热专家仍
  **20–29%**。即"内存越受限(正是流式最该用的场景),朴素缓存越崩,学习缓存越值"。
- 稳健性:learned 在 story/fact/code 三领域、全预算档**始终为正**优势(+6~+24pp)。
  code 生成专家复用更强(绝对命中率最高、小预算优势最大 +24pp)。
- 复现:`olmoe_trace <model> 64 16 <prompt>` 存 trace(NUTHATCH_TRACE_OUT),再
  `trace_replay <trace> 4 8 12 16 20 24 28 32`。

### 2026-07-11 — ★ 物理流式执行:OLMoE 在 1.2GB 而非 3GB 里跑,token-exact(P24)

- 机器/模型:M4 Pro;OLMoE-1B-7B Instruct Q4_K_M。
- 做法:StreamingModel 只常驻非专家权重;推理时逐 token 两段式(段A 路由→装槽
  [miss 真 pread]→段B 在槽上 mul_mat_id),每层有界槽缓存 capacity 个专家。
- 命令:`olmoe_stream <model> 12 16 The capital of France is`。
- 结果(峰值常驻内存 maximum RSS):

  | 路径 | 输出 | 峰值 RSS |
  |---|---|---|
  | 流式 capacity=16/层 | "…France is Paris." | **1203 MB** |
  | 常驻 olmoe_generate  | "…France is Paris." | 2975 MB |

- 结论:同一模型、**token-exact 同样输出**,流式常驻内存降到 ~40%(≈2.5×)。
  1.2GB ≈ 250MB 非专家 + 16 层×16 槽×~3MB 专家。capacity 越小越省(与命中率权衡,
  见 P22 曲线)。**至此 nuthatch 是真正的流式 MoE 引擎,非模拟器。**
- 代价:流式慢于常驻(逐 token 多趟小图 + miss 磁盘读);速度非本阶段目标。
- 正确性锚点:UT StreamingMatchesResident(流式==GreedyGenerateCached 逐 token 一致)。

### 2026-07-11 — pin/lru 配比曲线:OLMoE 上"越静态越好"(P27)

- 方法:固定总预算 16/层,把 learned-pin 的 pin 槽数从 0(≈纯 per-layer LRU)扫到
  16(全静态最热、无 LRU),看命中率。`trace_replay <trace> pin 16`。
- 结果(story trace,1072 records):pin=0 **48.0%** → pin=8 53.7% → pin=16 **57.8%**,
  **单调向上**,最优在 pin=budget(全静态)。
- 结论:OLMoE 专家使用不仅偏斜、而且**稳定**(热专家从头热到尾),所以追踪"最近
  使用"的 LRU 槽几乎不加值——静态钉住全局最热即最优。原本"最优在中间"的假设被否。
- ★ 诚实注脚:usage 直方图是从**被重放的同一条 trace** 建的(in-sample),pin=budget
  因而偏乐观。真实部署应从一次校准跑建直方图、用到**新输入**——那时 LRU 兜分布漂移
  会更有价值。"跨输入迁移"(prior 从 A 建、用到 B)是更严谨的后续问题。

### 2026-07-11 — 流式提速 2×:复用 compute buffer + 小图单线程(P28)

- 现象:流式每 token 有 ~34 趟小图(embed + 16 层×2 段 + final),原来每趟都
  ① ggml_init 新 malloc 一块 64MB compute buffer、② 以 4 线程 compute。T=1 的图
  极小,这两项都是纯开销(malloc churn + 578 次线程池 spin)。
- 改法:① 复用一块 thread_local compute buffer(用完即 ggml_free,顺序执行可共用;
  ggml 用外部 buffer 时不接管释放);② 极小图改单线程,只有末端 lm_head
  (n_vocab×n_embd 大乘)保留 4 线程。
- 结果(olmoe_stream 12 token,capacity 16,含加载):**2.73s → 1.36s(2.0×)**。
  输出不变、parity(StreamingMatchesResident)仍逐 token 一致。
- 未做(后续):批量/异步预取(miss 的磁盘读与 compute 重叠)、减少每 token 图趟数。

### 2026-07-11 — ★ 跨架构验证:learned 在 granite3-moe 上同样赢(P30)

- 动机:护城河结论不该是 OLMoE 专属。换一个**完全不同**的 MoE 架构验证。
- 模型:IBM **granite3-moe:1b**(`ollama pull`,~821MB,arch=granitemoe,**24 层 /
  32 专家 / 选 8**——与 OLMoE 的 16 层 / 64 专家迥异)。
- 方法:给 llama.cpp 的 eval-callback 打补丁,dump `ffn_moe_topk`(每层选中专家)成
  文本 trace(1629 records);nuthatch `trace_replay --text` 重放比三策略。**不把
  granite 跑进 nuthatch 引擎**——只验证缓存结论。提取法见 `docs/cross-arch-trace.md`,
  trace 附在 `data/granite3moe.trace.txt`。
- 结果:

  | 预算/层 (占 32) | learned | LRU | OS | learned−LRU |
  |---|---|---|---|---|
  | 4  (12.5%) | **22.4%** | 4.7% | 0.0% | **+17.7 pp** |
  | 8  (25%)   | **40.1%** | 30.4% | 30.2% | +9.7 pp |
  | 12 (37.5%) | **56.7%** | 47.7% | 45.2% | +9.0 pp |
  | 16 (50%)   | **71.6%** | 62.7% | 61.5% | +8.8 pp |

- 结论:**与 OLMoE 同规律**——learned 稳定赢 +8.8~17.7pp,预算越紧优势越大
  (12.5% 时 OS 塌到 0%)。**护城河跨架构成立,非 OLMoE 专属。**

---

## 数据点追加模板

```
### YYYY-MM-DD — <一句话标题>
- 机器/存储:
- 模型/量化:
- 命令:
- 结果:
- 结论:
```
