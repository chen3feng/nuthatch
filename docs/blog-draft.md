# 我从零写了一个流式 MoE 推理引擎,只为验证一个假设:该缓存哪些专家?

> nuthatch 建造手记 · 代码分析式复盘 · 初稿

## 缘起:一个被 colibrì 点燃的问题

前沿开源大模型的参数量已经越过 7000 亿。它们几乎都是 **MoE(Mixture-of-Experts)**:几十上百个"专家",但每个 token 只激活其中几个。这带来一个诱人的可能——**如果显存装不下全部专家,能不能只把常用的留在内存、其余留在磁盘按需读?**

[colibrì](https://github.com/JustVugg/colibri) 把这件事做到了极致:一个单文件 C 引擎,让 744B 的模型在 25GB 内存的笔记本上正确跑起来。我读完它的源码,冒出一个念头——它的专家缓存策略其实很朴素(每层 LRU + 可选固定热专家)。**那么:如果缓存策略本身能"学习"呢?用历史使用把最热的专家钉住,是不是能明显赢过 LRU?**

这就是 **nuthatch**(䴓,shī——一种囤食还记得藏在哪的小鸟)的护城河假设。为了验证它,我决定**从零造一个引擎**:原生读 GGUF、原生 k-quant、跨平台 C++、用 [blade-build](https://github.com/chen3feng/blade-build) 构建。目标很清楚:**涨 credit + 练手艺**。约束也很清楚:每步一个带 CI 和单测的 PR、引入依赖前先讨论、每加一层就对拍验证一次。

31 个 PR 之后,我可以给出两个有数据支撑的结论。先剧透:

- **假设成立,且跨架构成立**:在真实推理里,学习式缓存稳定赢 LRU/OS **6~24 个百分点**,而且**预算越紧优势越大**——不只在 OLMoE 上,换到 IBM 的 granite-moe 也一样。
- **引擎真的流式跑起来了**:同一个 OLMoE,同样逐 token 一致的输出,常驻内存从 3.0GB 降到 **1.2GB**。

下面按代码逐层拆。

---

## 全景:一条清晰的依赖链

nuthatch 的模块刻意分成三段,依赖单向流动:

```text
读格式          做研究              建引擎                    接护城河
gguf/  ──►  trace/  ──►  model/(olmoe_model,       ──►  expert_reader,
io/         cache/       attention, moe, forward,        expert_slot_cache,
moe/                     generate, kv_cache)             streaming_model,
                                                          streaming_forward
```

一个有意思的顺序决定:**我先做研究(trace + cache),再做引擎**。因为"学习缓存是否更好"这个问题,只需要**路由轨迹**(每 token 每层选了哪些专家)+ 一个缓存模拟器就能回答,根本不用先把引擎写出来。这让我在写第一行前向代码之前,就拿到了第一个研究信号。

一个 token 的生命周期(常驻路径):`token → get_rows 取 embedding → 16 层 × [注意力(+残差) → MoE FFN(+残差)] → output_norm → lm_head → argmax`。流式路径把这条链拆成两趟图,后面细说。

---

## 地基:读 GGUF,以及为什么是 pread 而不是 mmap

第一块地基是 [`src/gguf`](../src/gguf)(读元数据)和 [`src/io/tensor_source`](../src/io/tensor_source.cc)(读张量字节)。这里做了整个项目最早、也最关键的一个判断:**用 `pread` 定位读,而不是 `mmap`。**

理由直指护城河:流式缓存要**可控地预取和驱逐**专家。`mmap` 把这些全交给操作系统换页——而"操作系统页缓存"恰恰是我们想**打败**的基线(它就在 [`cache/os_page_cache`](../src/cache/os_page_cache_policy.h))。只有自己 `pread`,才谈得上"用学习策略决定谁留谁走"。

`TensorSource::ReadAt(offset, size, out)` 是一个带偏移、不改文件位置、可并发的定位读。后来(P29)它长出了 Windows 分支:POSIX 用 `pread`,Windows 用 `ReadFile + OVERLAPPED`,语义一致。

---

## 先做研究:不写引擎,先证明假设

[`src/trace`](../src/trace) 定义了一条紧凑的路由轨迹格式:每条记录是 `(token, layer, 选中的专家 ids)`。[`src/cache`](../src/cache) 则是研究的核心——一个 `CachePolicy` 接口 + 三种实现:

- `LruPolicy`:每层独立 LRU;
- `OsPageCachePolicy`:一个全局 LRU(模拟 OS 页缓存基线);
- `LearnedPinPolicy`:把**历史最热**的专家钉住常驻,其余走 LRU。

`Replay(trace, policy)` 用同一条轨迹重放每种策略,数出命中/缺失。这是整个研究"隔离变量、只比策略"的手法。第一个结果(在合成轨迹上)就显示 learned 赢——但那是合成的。真正的问题是:**在真实推理轨迹上呢?**

---

## 建引擎:让 OLMoE 真的算出 "Paris"

要拿真实轨迹,得先有能跑的引擎。[`src/model`](../src/model) 用 ggml 建图,逐块搭出 OLMoE:

- [`attention.cc`](../src/model/attention.cc):RMSNorm → QKV → **QK-norm**(OLMoE 对整段 n_embd 归一)→ 拆头 → **NEOX RoPE** → 打分/缩放/因果 mask/softmax → 加权 V → 输出投影。
- [`moe.cc`](../src/model/moe.cc):ffn_norm → router → softmax → `argsort_top_k` 选 8 → 取权重 → 分离的 up/gate `mul_mat_id` → SiLU → down `mul_mat_id` → 加权求和。
- [`forward.cc`](../src/model/forward.cc):把上面两块 × 16 层串起来,加残差,末端 output_norm + lm_head。

**正确性怎么保证?对拍 llama.cpp。** 这是全项目最重要的方法论:每加一层,就用 `llama-cli` 跑同样的输入,逐 token 比对。正是它抓到了一个关键 bug——

> 我第一次生成出的是 "The capital of France is **called** Paris",而 llama.cpp 是 "…is Paris"。对拍发现:OLMoE **不**对 top-k 权重做归一(`norm_topk_prob=false`)。改掉之后,首 token 精确一致(id 7785,` Paris`)。**对拍是最好的调试器。**

接着补齐**自包含**:[`tokenizer`](../src/tokenizer/tokenizer.cc) 实现了 GPT-2 byte-level BPE 的解码和编码。编码器要跑 OLMo 的预分词正则(带 `\p{L}` Unicode 属性 + 前瞻 `(?!\S)`)——这里有个依赖故事,后面讲。编码结果与 `llama-tokenize` **逐 id 一致**。至此,文本进、文本出,没有一行借 llama.cpp 做运行时。

再加 [`kv_cache`](../src/model/kv_cache.cc):decode 从每步重算整段(O(T²))降到每步 O(1)。ggml 实现里有个坑——持久缓存张量跨图调用容易踩"写缓存 vs 读缓存"的排序问题。我用 `concat`(读=旧缓存段拼新 k/v,天然有数据依赖)+ `cpy`(写=新段,和读区不相交)干净地绕开了它。正确性锚点:**带 cache 生成 == 不带 cache,逐 token 一致**。

---

## 护城河:从模拟到真实,再到跨架构

引擎能跑了,就能拿**真实轨迹**了。

**第一步(P19-P21):把真实推理接进 M2 策略。** [`expert_reader`](../src/model/expert_reader.cc) 按需从 GGUF `pread` 单个专家;[`expert_slot_cache`](../src/model/expert_slot_cache.cc) 是每层的有界槽缓存(LRU + 从盘装载)。然后我给前向加了个旁路:每层把 `argsort_top_k` 选中的专家 id 读出来,导成一条**真实推理的 RoutingTrace**。喂给 M2 的三种策略重放:

真实 OLMoE 上,learned-pin 稳定赢 LRU/OS **6~9pp**。M2 的模拟结论在真实推理路径上复现了。

**第二步(P22):曲线 + 跨领域稳健。** 我加了一个 model-free 的 `trace_replay`:读一条轨迹、秒级扫多个预算。跨 story/fact/code 三种 prompt,结论都成立,而且浮现出更强的规律——

| 每层预算(共 64 专家) | learned − LRU 优势 |
|---|---|
| 4 (6%)  | **+15 ~ +24 pp** |
| 8 (12%) | +9 ~ +16 pp |
| 16 (25%) | +6 ~ +11 pp |

**预算越紧,优势越大。** 在最紧的 6% 预算下,OS 全局缓存直接 **0% 命中**(彻底 thrash)、LRU 也只有 3~5%,而 learned 靠钉住热专家仍有 20~29%。这正是"内存越受限(流式最该用的场景),朴素缓存越崩,学习缓存越值钱"。

**第三步(P30):跨架构。** 一个自然的质疑:这会不会是 OLMoE 专属?于是我换了一个**完全不同**的 MoE——IBM 的 granite-moe:1b(24 层 / 32 专家,与 OLMoE 的 16 层 / 64 专家迥异)。nuthatch 引擎并不支持它的架构,但**我不需要把它跑进引擎**——只要它的路由轨迹。我给 llama.cpp 的 `eval-callback` 打了个小补丁,把名为 `ffn_moe_topk` 的张量 dump 成文本轨迹(提取法见 [`docs/cross-arch-trace.md`](cross-arch-trace.md),轨迹附在 [`data/granite3moe.trace.txt`](../data/granite3moe.trace.txt)),再喂 `trace_replay --text`:

| granite 预算/层(共 32) | learned | LRU | OS | learned − LRU |
|---|---|---|---|---|
| 4 (12.5%) | **22.4%** | 4.7% | 0.0% | **+17.7 pp** |
| 16 (50%)  | **71.6%** | 62.7% | 61.5% | +8.8 pp |

**同一规律,异构模型上成立。** 护城河不是 OLMoE 专属。

这里我否掉了自己的一个假设。我原以为 pin/lru 的最优配比"在中间"(钉一部分、留一部分给 LRU 兜临时热点)。但 [`SweepPinRatio`](../src/cache/trace_sweep.cc) 的曲线显示:OLMoE 上**越静态越好**,一路涨到 pin=budget。因为它的热专家太稳定了,LRU 的"最近使用"几乎不加值。**数据否掉直觉时,信数据**——同时我也如实标注了:这里的 usage 直方图是从被重放的同一条轨迹建的(in-sample),偏乐观;跨输入迁移是更严谨的后续问题。

---

## 物理流式:让"流式"名副其实

到这里,命中率结论已经拿到了——但引擎其实还**全常驻**(专家仍占着 ~4GB 内存)。研究结论不需要物理执行,但要让"流式"名副其实、真省内存,得再走一步。

**选择性加载(P23):** [`streaming_model`](../src/model/streaming_model.cc) 只把非专家权重(注意力/norm/router/embd/lm_head,约占模型一成)常驻,专家 3D 张量(约九成)留盘、交给 `expert_reader`。常驻内存从 ~4GB 降到 ~250MB。

**两段式前向(P24):** 这是全项目最难的一块。难点在于——**专家选择依赖运行时激活**,你必须先算完路由、知道选了哪几个专家,才能去磁盘装载、再计算。所以一层拆成两趟图:

- **段 A**:注意力(带 KV cache)+ 残差 + 路由 → 选中的全局专家 id,读回 host。
- **host**:`Ensure` 把选中的专家装进槽缓存(miss 真从盘 `pread`),并把全局 id 重映射成槽 id。
- **段 B**:在槽张量上 `mul_mat_id` 算专家 FFN + 残差。

逐 token 处理(T=1),活值极小,host 中转很便宜。正确性锚点照旧:**流式生成 == 常驻生成,逐 token 一致**([`StreamingForwardTest`](../src/model/streaming_forward_test.cc))。真机结果:

| 路径 | 输出 | 峰值内存 |
|---|---|---|
| 流式(每层缓存 16 专家) | "…is Paris." | **1.2 GB** |
| 常驻 | 同上,逐 token 一致 | 3.0 GB |

后来(P28)我把这条路径提速了 **2×**:每 token 有 ~34 趟小图,原来每趟都新 malloc 一块 64MB compute buffer、还以 4 线程 compute——对 T=1 的极小图纯属浪费。改成复用一块 buffer + 极小图单线程(省掉每 token 578 次线程池 spin)。

---

## 工程判断的记录

好项目的价值一半在"做了什么",一半在"为什么这么选、否掉了什么"。除了上面提到的 pread/mmap、norm_topk 发现、pin 曲线否定假设,还有几个:

- **C++ 而非 Go**:ggml 是 C/C++,MoE 前向是紧密的张量算子调用,跨 FFI 得不偿失。另一层动机——以往 C++ 引入外部模块一直是老大难,而 blade-build 恰好新支持了 vcpkg,正好拿这个新项目当**实战**,一举两得。
- **ggml 走 vcpkg,不自己写内核**:护城河在**算法**(缓存策略),不在**引擎**(矩阵乘 ggml 已做到极致)。复用成熟内核,把精力花在差异化上。
- **RE2 → PCRE2**:分词器正则要 Unicode 属性 + 前瞻。RE2 新版会拖进**庞大而模块众多的 abseil**,在 macOS 上撞 `CoreFoundation` 链接坑;PCRE2 自包含且支持前瞻,一次跑完整正则。"引入依赖前先讨论"这条规矩,这次省了大麻烦。
- **交叉验证选 granite 而非 26GB 的 Mixtral**:证明护城河不是 OLMoE 专属,不必硬跑 Mixtral(下载 26GB + 还得实现它的架构)。821MB 的 granite + 一个 llama.cpp 小补丁就够了——小下载、真异构,比再跑一个同类模型更有说服力。

还有一个隐形的功臣:**CI**。项目顺带 dogfood blade-build,几次被 Linux gcc 的 `-Werror` 抓到 macOS clang 放过的跨平台坑(`INFINITY` 缺 `<cmath>`、`memcpy` 缺 `<cstring>`、range-loop 绑定字符串字面量临时量)。**CI 是真防线。**

---

## 老实说:它还不能做什么

- 只有贪心 argmax,没有采样 / chat 模板 / batch。
- 流式路径仍慢于常驻(已提速 2×,但异步预取还没做)。
- 完整引擎构建走 blade(面向 Unix),Windows 目前只把可移植 I/O 层用真 MSVC 验证。
- 引擎只支持 OLMoE 架构;但**缓存研究已跨架构**(OLMoE + granite)。

---

## 尾声:一个假设,一条证据链

回头看,这个项目其实是一条围绕单个假设展开的证据链:

**"用历史使用学出来该缓存哪些专家,会明显赢过朴素缓存"** — 从合成轨迹(第一个信号)→ 真实 OLMoE 推理(6~9pp)→ 跨领域稳健 + 预算越紧越赢(6~24pp)→ 跨架构(granite,+8.8~17.7pp)。每一环都有可复现的数据,每一步引擎的正确性都用对拍或 parity 锚住。

对我而言,收获是双份的:一个能自证的研究结论,和一次把推理引擎从 GGUF 解析到 MoE 前向到物理流式亲手过一遍的经历。如果你也想学推理引擎,这个仓库是按学习顺序长出来的——顺着 [31 个 PR](https://github.com/chen3feng/nuthatch/issues/1) 读,每一步都有"为什么"。

*(初稿。源码、命中率曲线、内存实测均在 [chen3feng/nuthatch](https://github.com/chen3feng/nuthatch)。)*
