---
title: 首页
nav_order: 1
---

# nuthatch 源码分析

[nuthatch](https://github.com/chen3feng/nuthatch)(䴓,shī——一种把种子囤进树皮缝、还记得藏在哪的小鸟)是一个**从零手写的流式 MoE 推理引擎**:原生读 GGUF、原生 k-quant、跨平台 C++、用 [blade-build](https://github.com/chen3feng/blade-build) 构建。

它不是又一个"能跑通就发"的玩具,而是围绕**一个假设**展开的证据链:

> MoE 模型有几十上百个专家,每个 token 只用其中几个。如果显存装不下全部专家,该把谁留在内存里?**用历史使用"学"出该缓存哪些专家,能明显赢过朴素的 LRU / OS 页缓存吗?**

32 个 PR 之后,答案是**能,且跨架构成立**:真实推理里学习式缓存稳定赢 6~24 个百分点,预算越紧优势越大——不只在 OLMoE,换到 IBM 的 granite-moe 也一样。与此同时,引擎**真的**流式跑起来了:同一个 OLMoE、逐 token 一致的输出,常驻内存从 3.0GB 降到 1.2GB。

本系列从源码出发,按**建造顺序**逐模块拆解:读格式 → 先做研究 → 建引擎 → 接上护城河 → 物理流式落地。每个论断都链接到 `src/` 里的精确文件(在 VS Code 中 `Cmd/Ctrl + 点击`可跳转)。

## 文档

| # | 文档 | 内容 |
|---|------|------|
| 1 | [架构概述与设计哲学](01-架构概述) | 假设是什么、三段式依赖链、一个 token 的生命周期、关键设计决策一览 |
| 2 | [代码结构、构建与 CI](02-构建与CI) | blade-build 工程、经 vcpkg 引 ggml、每步一 PR、dogfood 构建系统、CI 抓到的跨平台坑 |
| 3 | [读 GGUF 与定位读](03-GGUF与定位读) | GGUF 元数据/张量布局、`pread` 而非 `mmap`(为打败 OS 页缓存)、跨平台 I/O |
| 4 | [研究先行:trace 与缓存策略](04-研究先行) | 不写引擎先做研究、路由 trace 格式、LRU/OS/learned-pin、同轨迹重放隔离变量 |
| 5 | [OLMoE 前向:注意力、MoE、ggml 建图](05-前向) | QK-norm/NEOX RoPE 的注意力、`mul_mat_id` 的 MoE、整图前向 |
| 6 | [能推理:贪心生成、对拍与 norm_topk 发现](06-能推理) | 贪心生成、逐 token 对拍 llama.cpp、`norm_topk=false` bug 的抓捕 |
| 7 | [自包含分词器:BPE + PCRE2](07-分词器) | byte-level BPE 解码/编码、OLMo 预分词正则、RE2→PCRE2 的依赖抉择 |
| 8 | [KV cache 与图排序陷阱](08-KVcache) | O(T²)→O(1)、持久缓存张量、`concat`+`cpy` 避开 ggml 图排序坑、parity |
| 9 | [护城河:命中率、曲线、跨领域、跨架构](09-护城河) | 真实推理 trace、learned 赢的曲线、pin 曲线否定假设、granite 跨架构验证 |
| 10 | [物理流式:选择性加载与两段式前向](10-物理流式) | 非专家常驻/专家留盘、两段式图编排、1.2GB vs 3GB、提速 2× |
| 11 | [工程判断与适用边界](11-边界) | pread/mmap、C++/Go、依赖抉择、granite 而非 Mixtral 等判断,以及诚实的边界 |

## 快速开始

```bash
git clone https://github.com/chen3feng/nuthatch.git
cd nuthatch
blade test //...     # 需要 blade-build + ninja + vcpkg
```

源码在同仓 `src/` 下。在 VS Code 中打开,文档里的代码引用 `Cmd/Ctrl + 点击`即可跳转。
