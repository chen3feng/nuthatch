# nuthatch

> **一个用 C++ 从零写的 GGUF MoE 流式推理引擎 —— 该缓存的缓存,该留盘的留盘。**

**nuthatch**(䴓,shī)是一种把种子塞进树皮缝里囤着、还记得藏在哪、头朝下爬树的小鸟。代号取自它的两个本事——**囤食 + 记住位置**——正是这个项目的护城河:**用「使用历史」学出来的逐专家流式缓存**。

---

## TL;DR:两个可以拿去讲的硬结果

这不是又一个"能跑通就发"的玩具。它围绕一个问题做出了**可复现的证据**:

> MoE 模型有几十上百个专家,每个 token 只用其中几个。如果显存装不下全部专家,该把谁留在内存里?

**① 学习缓存在真实推理上稳定赢朴素缓存(护城河)**
拿真实 OLMoE 推理导出的专家选择轨迹,对比三种缓存策略的命中率(miss = 要从磁盘读)。跨故事/事实/代码三种 prompt 都成立:

| 每层预算(共 64 专家) | learned-pin | LRU | OS 页缓存 | learned 优势 |
|---|---|---|---|---|
| 4 (6%)  | **20–29%** | 3–5% | **0%** | **+15 ~ +24 pp** |
| 8 (12%) | **31–48%** | 22–32% | 22–32% | +9 ~ +16 pp |
| 16 (25%) | **54–73%** | 46–63% | 47–63% | +6 ~ +11 pp |

**预算越紧,优势越大**——内存越受限(正是流式最该用的场景),朴素 LRU/OS 越是崩溃,学出来的缓存越值钱。

**② 引擎真的流式跑起来了,不是模拟(工程)**
专家不常驻,推理时按需从磁盘装进有界槽缓存再算:

| 路径 | 输出 | 峰值内存 |
|---|---|---|
| **流式**(每层缓存 16 专家) | "The capital of France is Paris." | **1.2 GB** |
| 常驻(全部专家进内存) | 同上,逐 token 一致 | 3.0 GB |

**同一模型、逐 token 完全一致的输出,内存降到 ~40%。**

---

## 起因与目的

看到 [colibrì](https://github.com/JustVugg/colibri)——一个把 MoE 专家留在磁盘、按需流式读入来"用小内存跑大模型"的单文件 C 引擎——觉得思路很妙,但它自造模型格式、只跑 CPU、缓存策略也简单。于是想:**自己写一个,原生读 GGUF、原生 k-quant 量化,并把「缓存策略」做成真正的差异化。**

目的很直白,两条:

1. **涨 credit** —— 做一个有真实数据支撑、别人愿意看的东西;
2. **练手艺** —— 把推理引擎从 GGUF 解析到 MoE 前向到流式缓存,每一层都亲手过一遍。

约束也定得很清楚,后面每一步都守住了:**每步一个独立 PR(带 CI + 单测 + 详细说明)**、**引入任何第三方库前先讨论**、辅助脚本用 Python、主体用 C++、顺便拿这个项目 **dogfood 作者自己的构建系统 [blade-build](https://github.com/chen3feng/blade-build)**。

护城河从一开始就想清楚了:**不是"引擎"**(那是 ggml 已经解决的 commodity),**而是"用使用历史学出来该把哪些专家常驻"这个算法**,以及围绕它的可复现研究。

---

## 快速开始

### 构建

```bash
# 需要 blade-build + ninja + vcpkg(设好 $VCPKG_ROOT)
blade test //...          # 全部单测(22 个,CI 每 PR 必过)
blade build //src/model:olmoe_generate    # 单个目标
```

第三方依赖只有三个,全走 vcpkg,且都是**引入前先讨论过**的:`ggml`(推理内核)、`gtest`(测试)、`pcre2`(分词器正则)。

### 用:四个工具

模型用 [OLMoE-1B-7B Instruct 的 Q4_K_M GGUF](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF)(16 层、64 专家选 8、~4GB)。

> **拿模型的一个便捷办法**:装了 [Ollama](https://ollama.com) 的话,`ollama pull` 下来的就是 GGUF,直接在它的缓存目录里找到喂给 nuthatch——
> `~/.ollama/models/blobs/sha256-*`(最大的那个 blob 就是权重 GGUF),省得再单独下载一份。

```bash
# ① 常驻推理:文本进 → 自己分词 → KV cache 生成 → 自己解码 → 文本出
olmoe_generate <model.gguf> 24 The capital of France is
#   → The capital of France is Paris.

# ② 物理流式:专家按需从盘装进有界槽缓存,真省内存(输出与 ① 逐 token 一致)
olmoe_stream <model.gguf> 12 16 The capital of France is
#   → text: ...Paris.   peak RSS: 1203 MB   (16 专家/层)

# ③ 护城河证据:真实推理导出路由 trace,三策略比命中率
olmoe_trace <model.gguf> 64 16 Once upon a time

# ④ 曲线:读一条 trace 秒级扫多个预算(不重载模型)
trace_replay some.trace 4 8 12 16 24 32
```

全程没有一行借 llama.cpp 做运行时——分词、KV cache、MoE 前向、流式缓存都是自己的代码;正确性靠**逐 token 对拍 llama.cpp** 保证。

---

## 沿着 24 个 PR 学

这个仓库是**按学习顺序**长出来的。想学推理引擎,可以顺着 PR 读——每个 PR 都有详细的"为什么这么做",代码有简洁注释,且都带单测。按里程碑分组:

| 阶段 | PR | 你会学到 |
|---|---|---|
| **M0 脚手架** | #2–#3 | blade-build 工程结构、CI、gtest、怎么把 ggml 经 vcpkg 引进来 |
| **M1 读 GGUF** | #4–#6 | GGUF 元数据/张量布局、**为什么用 `pread`+`fadvise` 而非 `mmap`**、MoE 融合专家张量的内存排布 |
| **M2 先做研究** | #7–#10 | **不写引擎先做缓存研究**:路由 trace 格式 → LRU / OS 页缓存基线 / usage-learned pin,用同一条 trace 重放对比(隔离变量)→ **第一个结果** |
| **M3 能推理** | #11–#15 | ggml 建图:OLMoE 加载 → 注意力(MHA + QK-norm + NEOX RoPE)→ MoE FFN(`mul_mat_id`)→ 整图前向 → 贪心生成 → **对拍 llama.cpp 通过** 🎉 |
| **自包含** | #16–#17 | GPT-2 byte-level BPE 分词器:解码(byte↔unicode 反查)+ 编码(**PCRE2 跑 OLMo 预分词正则** + BPE 合并),编码与 `llama-tokenize` 逐 id 一致 |
| **KV cache** | #18 | 单序列 KV cache 的 ggml 实现,decode 从 O(T²) 降到每步 O(1),**用 `concat` 避开图排序陷阱** |
| **流式护城河** | #19–#22 | 按需读单个专家(`pread`)→ 有界槽缓存(LRU + 从盘装载)→ **把真实推理 trace 接进 M2 策略** → 命中率曲线 + 跨领域稳健性 |
| **物理流式** | #23–#24 | 选择性加载(非专家常驻、专家留盘)→ **逐 token 两段式前向**(段A 路由 → 装槽 → 段B 在槽上算)→ 真省内存、token-exact |

每一步都有一个"正确性锚点":`能推理`对拍 llama.cpp、`tokenizer`对拍 llama-tokenize、`KV cache`和`流式`都用 **`==` 常驻路径逐 token 一致**的单测守住。这套"每加一层就锚一次正确性"的做法,本身就是最值得学的东西。

---

## 模块结构(读代码从这里入手)

```text
src/
  gguf/       GGUF 元数据读取器(P3)
  io/         张量字节读取:pread,非 mmap(P4)
  moe/        MoE 融合专家张量布局(P5)
  trace/      路由 trace 二进制格式 + 读写(P6)
  cache/      缓存策略:cache_policy 接口 · lru · os_page · learned_pin
              · trace_sweep/replay(P8–P10, P22)—— 护城河的算法在这里
  tokenizer/  GPT-2 byte-level BPE:解码 + PCRE2 预分词 + BPE 编码(P16–P17)
  model/      引擎主体(P11–P24):
              olmoe_model   加载(config + 常驻权重)
              attention     注意力块(+ 带 KV cache 的变体)
              moe           MoE FFN 块
              forward       整图前向(+ 带 KV cache 的变体)
              generate      贪心生成(+ KV cache + 导出真实路由 trace)
              kv_cache      单序列 KV 缓存
              expert_reader     按需从 GGUF pread 单个专家
              expert_slot_cache 每层有界专家槽缓存(LRU + 装载)
              streaming_model   显存受限加载(非专家常驻)
              streaming_forward 两段式物理流式前向
```

一条清晰的依赖链:`gguf/io/moe`(读格式)→ `trace/cache`(做研究)→ `model`(建引擎)→ `expert_reader/slot_cache/streaming_*`(接上护城河)。

---

## 开发中的抉择(工程判断的记录)

好项目的价值一半在"做了什么",一半在"为什么这么选、以及否掉了什么"。几个关键路口:

- **选 GGUF 格式(而非自造)**:GGUF 是本地大模型推理的**事实标准**(llama.cpp / ggml 生态)。选它等于**站在生态上**:① HuggingFace 上现成的量化模型随手可得、`ollama pull` 直接下,不用自己转换;② ggml **原生读 GGUF、原生 k-quant**,不必重写量化内核;③ 工具链成熟,`llama-tokenize` / `llama-cli` 正好当**对拍基准**验证正确性。colibrì 自造格式省了依赖却也断了生态,nuthatch 反过来押注普及度与库支持。
- **`pread` + `posix_fadvise` 而非 `mmap`**:流式缓存要**可控地预取/驱逐**专家,`mmap` 把这些交给 OS 换页——正是我们想打败的 baseline(它就在 `cache/os_page_cache`)。自己 `pread` 才能实现学习策略。
- **C++ 而非 Go**:ggml 是 C/C++,MoE 前向是紧密的张量算子调用,跨 FFI 得不偿失。Go 适合写辅助工具,主体留 C++。另一层动机:以往 C++ 引入外部模块一直是老大难,而 blade-build 恰好新支持了 vcpkg——正好拿这个新项目当**实战**,一举两得地把这条链路走通。
- **ggml 走 vcpkg,不自己写内核**:护城河在**算法**(缓存策略),不在**引擎**(矩阵乘法 ggml 已经做到极致)。复用成熟内核,把精力花在差异化上。
- **RE2 → PCRE2**:分词器预分词要跑 OLMo 的正则(带 `\p{L}` Unicode 属性 + 前瞻 `(?!\S)`)。RE2 新版会拖进**庞大而又模块众多的 abseil**,在 macOS 上撞 `CoreFoundation` 链接坑;PCRE2 **自包含、且支持前瞻**,能一次跑完整正则。**"引入依赖前先讨论"这条规矩,这次省了大麻烦。**
- **`norm_topk=false` 的发现**:第一次生成出 "…is **called** Paris",而 llama.cpp 是 "…is Paris"。对拍发现 OLMoE **不**对 top-k 权重归一;改掉后首 token 精确一致。**对拍是最好的调试器。**
- **命中率:先 trace + 重放,不急着物理执行**:要证明"learned 赢",不需要真把专家搬进显存算——真实推理导出访问轨迹、用模拟器重放对比即可(标准方法)。**先拿到研究结论,物理执行作为后续工程。**
- **两段式流式前向**:专家选择依赖运行时激活,所以必须**先算路由拿到选中的专家、再装载、再计算**——一层拆成两趟图。读旧缓存段用 `concat`(天然有数据依赖)、写新段用 `cpy`(读写区不相交),干净地避开了 ggml 的图排序陷阱。

这些判断连同 `docs/blade-feedback.md`(dogfood 构建系统时的真实反馈,包括 CI 抓到的 `-Werror` 跨平台坑)一起,是这个项目"教学价值"的另一半。

---

## 老实说:它还不能做什么

保持诚实(这也是项目一贯的态度):

- **只有贪心 argmax**,没有采样、没有 chat 模板、没有 batch/并发。
- **流式路径慢**(逐 token 多趟小图 + miss 时磁盘读)——本阶段目标是"**真的**在受限内存里跑起来且 token-exact",不是速度。
- **验证在 CPU**;ggml 的 Metal/CUDA/Vulkan 后端理论可接但未验证。
- **只支持 OLMoE 架构**(可作为接更多 MoE 的模板)。

---

## 文档

- [`docs/ROADMAP.md`](docs/ROADMAP.md) — 分步实现计划(Master Issue #1),每个 PR 引用它
- [`docs/dependencies.md`](docs/dependencies.md) — 第三方依赖账本(引入前先讨论)
- [`docs/blade-feedback.md`](docs/blade-feedback.md) — dogfood blade-build 的真实反馈
- [`bench-log.md`](bench-log.md) — 研究日志:命中率曲线、内存实测等硬数据

## License

Apache 2.0(拟)。
