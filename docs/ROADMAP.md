# nuthatch — 路线图 / Master Issue

> **一个支持 GGUF、跨 Mac/Linux/Windows、用 blade-build 构建的 C++ MoE 流式推理引擎。**
> 护城河不是"引擎"(那是 ggml 已解决的 commodity),而是 **usage-learned 逐专家流式缓存**这个算法。
> 本项目同时是 **blade-build 的真实 dogfooding**:含 C 库依赖、跨平台、多后端、gtest UT 的中等复杂工程。

面向**学习**:每一步一个独立 PR,小、可测、CI 绿、带 UT。按顺序做,每个 PR 只教一件事。

---

## 目标 / 非目标

**目标**
- 原生**读** GGUF(不转格式)、原生 k-quant(靠 ggml)、多后端(CPU/Metal/CUDA/Vulkan via `ggml-backend`)。
- 路由专家**留盘按需流式**,配 LRU / pin / **usage-learned** 缓存 + 预取。
- 跨 Mac/Linux/Windows,blade-build 构建,gtest 单测,GitHub Actions CI。
- 产出可复现的**研究结论**:learned cache vs LRU vs OS 页缓存的命中率曲线;MPS×命中率曲线。

**非目标(至少 v1)**
- 不做通用多架构大而全(先 OLMoE/Mixtral 两个标准 MoE + 一个 MLA 族)。
- 不重写 k-quant / 后端(用 ggml)。
- 不追求 SOTA 吞吐——这是研究/学习引擎,不是生产引擎。

---

## 架构决定(定调,后续 PR 遵守)

1. **建在 ggml(低层)之上,不建在 llama.cpp 之上**。llama.cpp 假设权重全 mmap 常驻,做逐专家流式是跟它核心假设对着干。ggml 给张量 + k-quant + `ggml-backend`(多后端免费)。llama.cpp 的 `src/llama-arch.*` / `llama-graph.*` / `llama-model.*` 仅作**图构建参考**。
2. **依赖 ggml**:先用本地源码 `/Volumes/code/exernal/ai/ggml`(blade `cc_library` 包一层或走 submodule);发布时切 vcpkg `ggml[metal|cuda|vulkan]`(v0.11.1,MIT)。
3. **最小外科切入点**:自定义 ggml **backend buffer**,让专家张量按需 fault-in,其余复用 ggml。
4. **差异化 = 学习缓存**,所以研究核心(trace 重放模拟器,M2)**先于**完整引擎——它几乎不占盘,又直接产出 credit 结论。

---

## 通用开发约定(每个 PR 都遵守)

- **一个 PR 一件事**,能独立 review、独立 CI 绿。
- 分支名 `pN-<slug>`(如 `p3-gguf-reader`),PR 标题 `[PN] <做什么>`。
- **每个 PR 必须带 UT**(gtest),覆盖新增逻辑的正常 + 边界路径。
- **每个 PR 必须 CI 绿**(workflow 在 PR#1 建立,之后所有 PR 复用)。
- **blade dogfooding**:凡在 blade 上踩到的坑(第三方 C 库链接、平台条件编译、CI 集成、错误信息不清等),记到 `docs/blade-feedback.md`,能修的顺手给 blade-build 提 issue/PR。
- 代码风格:跟 blade-build / Google C++ 一致;每个 `cc_library` 配同名 `cc_test`。
- **详细 PR 说明**:每个 PR 按下面模板写清"为什么/做了什么/怎么测/学到什么",既是 review 材料,也是学习笔记。
- **代码注释**:**简洁而清晰**——注释解释"为什么这么做 / 非显然的约束"(如 super-block 对齐、为何用 pread 不用 mmap、fault-in 时机),**不复述代码在做什么**。关键数据结构和跨平台分支必须有一句话说明。
- **辅助脚本/工具用 Python**:运行时引擎是 C++,但**离线/辅助工具用 Python**(和 colibrì 一样的分工)——fixture 生成、trace 抓取与解析、画图、benchmark 编排、GGUF 探查。Python 工具放 `tools/`,不进运行时依赖。

### PR 说明模板
```markdown
## [PN] 标题
### 为什么 (Why)
这个 PR 解决什么、在路线图里的位置、依赖哪个 PR。
### 做了什么 (What)
新增/改动的模块、关键设计选择(为何这么选)、非显然的取舍。
### 怎么测 (Tests)
新增的 UT 覆盖了哪些正常/边界路径;如何本地复现(`blade test //...`)。
### 学到什么 (Notes)
这个 PR 涉及的知识点(ggml API / GGUF 布局 / 缓存策略 / blade 用法…)。
### blade 反馈 (if any)
踩到的 blade 摩擦点(同步到 docs/blade-feedback.md)。
```

### blade BUILD 约定(示例)
```python
# BLADE_ROOT 在仓库根

# src/gguf/BUILD
cc_library(
    name = 'gguf',
    srcs = ['gguf_reader.cc'],
    hdrs = ['gguf_reader.h'],
    deps = ['//thirdparty/ggml:ggml'],
)
cc_test(
    name = 'gguf_test',
    srcs = ['gguf_reader_test.cc'],
    deps = [':gguf', '//thirdparty/googletest:gtest_main'],
    testdata = ['//testdata:tiny_moe.gguf'],
)
```
构建/测试:`blade build //src/...` · `blade test //src/...`

---

## CI workflow 规范(PR#1 建立,之后不再改结构)

`.github/workflows/ci.yml`:
- 触发:`push` / `pull_request`。
- **matrix**:`ubuntu-latest` + `macos-latest`(Windows 在 PR#16 加入 matrix)。
- 步骤:checkout → 装 python3 + ninja + 编译器 → 装 ggml(本地 build 或 vcpkg)→ 装 blade(`pip install blade-build` 或按其 README)→ `blade test //...`。
- 产物:测试通过/失败;后续 PR 可加 `blade test --coverage`。

---

## 里程碑总览

| M | 主题 | PR | 产出 |
|---|------|----|------|
| **M0** | 工程骨架(toolchain 绿) | #1–#2 | blade+CI+gtest+ggml 链接通 |
| **M1** | GGUF 读取 | #3–#5 | 能读 GGUF 元数据 + 专家张量切片 |
| **M2** | 路由 trace + 缓存模拟器 ★研究核心 | #6–#10 | **第一张命中率曲线图(learned vs LRU vs OS)** |
| **M3** | 真实流式专家存储 + 引擎 | #11–#14 | token-exact 的流式 MoE 前向 + e2e 命中率 |
| **M4** | 平台 & 计算后端 | #15–#17 | Metal / Windows / 预取+MTP |

---

## PR 清单(逐个)

### M0 — 工程骨架

**PR#1 — blade 脚手架 + CI + gtest 冒烟**
- 交付:`BLADE_ROOT`、`thirdparty/googletest`、一个 trivial `cc_library`(如 `util:version`)+ `cc_test`、`.github/workflows/ci.yml`。
- UT:`version_test` 断言版本号。
- CI:linux+mac 跑 `blade test //...` 绿。
- 完成:CI 徽章绿;`blade test` 本地通过。
- 学到:blade BUILD 结构、gtest、GH Actions。

**PR#2 — 链接 ggml**
- 交付:`thirdparty/ggml/BUILD`(包本地 ggml 源码或 submodule),一个调用 ggml 的冒烟库。
- UT:`ggml_smoke_test` —— `ggml_init` → 建一个 `a*b` 的 1×1 图 → 跑 CPU backend → 断言结果。
- CI:linux+mac 都能链接 + 跑通。
- 完成:ggml 作为 blade 依赖可用。
- 学到:blade 链接第三方 C 库(**首个 dogfooding 重点**)、ggml `ggml_context` / `ggml_backend` 基础。

### M1 — GGUF 读取

**PR#3 — GGUF 元数据读取器**
- 交付:`GgufReader`,包 ggml `gguf.h`:打开文件 → 列出张量(name/type/offset/shape)+ KV 元数据(arch、n_layers、n_expert…),**不加载数据**。
- UT:对一个 tiny GGUF fixture 断言 arch / 层数 / 专家数 / 张量数。
- 完成:能打印任意 GGUF 的结构。
- 学到:GGUF 格式、ggml gguf API。

**PR#4 — 张量字节读取(pread,非 mmap)**
- 交付:`TensorSource` 抽象 + `pread` 实现:按 PR#3 的 offset 读某张量的原始字节。
- UT:读一个已知张量,checksum 对比 ggml 自己读的结果。
- 完成:定位读通;确立"非 mmap"设计。
- 学到:`pread`、张量 offset、反 mmap 的理由。

**PR#5 — MoE 专家张量布局**
- 交付:识别专家张量(`blk.N.ffn_{gate,up,down}_exps.weight`),解析融合 3D 布局,算**每专家字节切片** + **super-block 对齐检查**。
- UT:对 OLMoE/Mixtral fixture 枚举专家,断言切片 offset/size 对齐 256 元素块。
- 完成:能列出"第 L 层第 E 个专家在文件哪段"。
- 学到:GGUF 里 MoE 的存法、流式切片的对齐约束。

### M2 — 路由 trace + 缓存模拟器 ★研究核心(几乎不占盘)

**PR#6 — 路由 trace 格式 + I/O**
- 交付:紧凑格式 `(token_idx, layer, [expert_ids])` 的 reader/writer。
- UT:round-trip;损坏输入的错误处理。
- 学到:trace 设计。

**PR#7 — 从 llama.cpp 抓真实 trace**
- 交付:一个小 patch/hook(或解析 llama.cpp 调试输出),对 OLMoE 真实生成 dump 每 token 的路由,产出真 trace fixture。
- UT:golden trace 的形状/不变量。
- 学到:真实路由长什么样、给 llama.cpp 插桩。

**PR#8 — 缓存策略框架 + LRU + 重放器**
- 交付:`CachePolicy` 接口(admit/evict/on_hit)+ 每层 LRU 实现 + 重放引擎(把 trace 喂给 N 槽/层的缓存,报**命中率 + 读字节**)。
- UT:构造 trace → 已知命中率。
- 学到:缓存策略抽象、指标定义。

**PR#9 — OS 页缓存基线 + 纯 LRU 对照**
- 交付:建模 "mmap+页缓存"基线(全局按页/专家 LRU),与每层 LRU 对比。
- UT:在同一 trace 上断言相对行为。
- 学到:你要打败的基线到底多强。

**PR#10 — ★ usage-learned pin 策略(差异化核心)**
- 交付:持久化每专家使用直方图,"启动"时 pin 最热的一批;对比 LRU / OS 的命中率提升。
- UT:偏斜 trace 上,learned-pin 命中率 > LRU 一个既定 margin。
- 完成:**产出第一张研究图(命中率 ~ RAM 预算,三条策略)。**
- 学到:差异化算法本身 + 你的第一个可发布结论。

### M3 — 真实流式专家存储 + 引擎

**PR#11 — 专家存储(真实 I/O)**
- 交付:从 GGUF pread 专家切片(super-block 对齐)+ 合成 blob 模式(纯 IO 基准)。跨平台 I/O 抽象(Linux `pread`+`fadvise` / macOS `F_NOCACHE`)。
- UT:读专家切片 checksum;IO 模式带宽 sanity。
- 学到:真实流式 I/O、跨平台绕缓存。

**PR#12 — ★ 自定义 ggml backend buffer(fault-in,技术风险最高)**
- 交付:实现一个 `ggml_backend_buffer`,访问专家张量时**惰性加载**切片(背后接 PR#11 存储 + PR#8 策略)。
- UT:一个读专家张量的 ggml 图触发加载,结果与"全常驻"读一致。
- 完成:核心机制打通——**单独隔离这个 PR,风险集中**。
- 学到:ggml backend 内部、整个方案的命门。

**PR#13 — 最小 MoE 前向(单架构)**
- 交付:复用 llama.cpp 的 OLMoE/Mixtral 图(或在 ggml 上搭最小 MoE 前向),专家由流式 buffer 支撑。对齐 llama.cpp 的 logits(几 token,token-exact)。
- UT:logits 与参考在容差内一致。
- 学到:ggml 图构建、MoE 前向、正确性 harness。

**PR#14 — 学习缓存接入真实前向**
- 交付:端到端 tok/s + 命中率,LRU vs learned-pin 在真模型上对比。
- UT:e2e 冒烟(短生成),指标输出。
- 完成:模拟器结论在真机复现。
- 学到:拼起来、真实测量。

### M4 — 平台 & 计算后端

**PR#15 — Metal 后端 + offload 扫描**
- 交付:启用 `ggml-metal`;测 Apple 上 ngl/命中率交互(那条 MPS×命中率曲线)。
- UT:有 Metal 时选 Metal,否则回退 CPU。
- 学到:`ggml-backend` 选择、MPS×命中率曲线。

**PR#16 — Windows I/O 路径**
- 交付:`ReadFile` + `OVERLAPPED` + `FILE_FLAG_NO_BUFFERING`,藏在跨平台存储接口后;CI matrix 加 `windows-latest`。
- UT(Windows CI):存储读出正确字节。
- 学到:Windows 异步 I/O、真·跨平台。

**PR#17(拉伸)— 预取 + MTP 投机**
- 交付:router-lookahead 预取(专用 I/O 线程)+ MTP 批量验证(如果目标模型有 MTP 头)。
- UT:预取预测命中率;MTP 批量验证 token-exact。
- 学到:colibrì 剩下的两个杠杆。

---

## blade-build 反馈(dogfooding,贯穿全程)

单列一个 `docs/blade-feedback.md`,随做随记。预期会压到的点:
- **第三方 C 库链接**:ggml 的 include 路径、平台库(`-framework Metal`、`-lcudart`)、条件依赖(Metal 仅 mac、CUDA 仅有卡)——blade 的 `cc_library` 表达是否顺手?
- **平台条件编译**:按 `OSTYPE`/后端选不同 srcs/deps(PR#11/#15/#16)——blade 的条件写法够不够清晰?
- **CI 集成**:blade 在 GH Actions(linux+mac+win)上装 + 跑的摩擦、增量构建缓存。
- **gtest 集成**:`cc_test` + `testdata` + 覆盖率的体验。
- **错误信息**:依赖缺失/循环依赖时 blade 的报错是否好懂。
- 每条:现象 → 期望 → 是否可修 → 是否已给 blade 提 issue/PR。

---

## 依赖关系(哪些能并行)

```
#1 → #2 →┬→ #3 → #4 → #5 ─────────────┐
         │                             ├→ #11 → #12 → #13 → #14 →┬→ #15
         └→ #6 → #7 → #8 → #9 → #10    ┘(#10 的结论指导 #14)      ├→ #16
             (M2 可与 M1 后半并行)                                 └→ #17
```
- M2(#6–#10,研究核心)在 #5 后即可与 M3 前置并行——它不依赖完整引擎。
- 建议顺序:**先 M0→M1→M2 拿到第一张图**(便宜、出成果、纯学习),再回头做 M3 的真实引擎。
