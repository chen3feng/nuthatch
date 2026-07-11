---
title: KV cache 与图排序陷阱
nav_order: 9
---

# KV cache:O(T²)→O(1) 与 ggml 图排序陷阱

## 1. 问题:每步重算整段

朴素的 `GreedyGenerate` 每生成一个 token 就重跑一次**整段**前向——decode 第 t 步要重算 t 个 token 的注意力和 MoE。总复杂度 O(T²),而且在长序列上还会因为图 ctx 内存不够而崩。KV cache 是标准解法:把每层已算过的 K/V 缓存下来,decode 每步只算**新** token 的 Q/K/V,对缓存的 K/V 做注意力——每步 O(1)。

## 2. KvCache:常驻、跨步存活

[`kv_cache.cc`](../../src/model/kv_cache.cc) 给每层持有 K/V 缓存张量 `[n_embd_kv, max_seq]`(f32),放在**自己的持久 ggml_context** 里,跨 decode 步存活。每步把新 token 的 k/v 写进 `[n_past : n_past+T]` 段。

## 3. 陷阱:ggml 的图排序

这里有个 ggml 的坑。持久缓存张量跨图调用,最容易踩"**写缓存 vs 读缓存**"的排序问题:如果这一步既要把新 k/v 写进缓存、又要读缓存做注意力,而 ggml 的依赖是按张量 src 指针建的——一个"缓存的 view"并不会自动依赖于"写进缓存的 cpy",于是执行顺序无保证,可能读到还没写的、或写乱。

llama.cpp 用它自己的一套 backend scheduler + 精心的图构造处理这个。nuthatch 单序列,用一个更简单、可证明正确的办法(见 [`attention.cc`](../../src/model/attention.cc) 的 `BuildAttentionCached`):

- **读** = `concat(cache[0:n_past], 新 k/v)`——`concat` 显式依赖新 k/v,排序天然正确;
- **写** = `cpy(新 k/v → cache[n_past:n_past+T])`——供**下一步**读;
- 本步的**读区 `[0:n_past]`(旧数据)与写区 `[n_past:]`(新区)不相交** → 步内没有先后依赖,不需要同步。

因果 mask 用 `ggml_diag_mask_inf(kq, n_past)` 的偏移形式:新 token 在绝对位置 `n_past+i`,只能看到 key `≤ n_past+i`。RoPE 的 k 存**旋转之后**(按绝对位置),缓存里就是 post-RoPE 的 K——与 llama.cpp 一致。

## 4. 正确性锚点:cached == uncached

KV cache 是纯优化,不该改变任何输出。单测 [`CachedMatchesUncached`](../../src/model/generate_test.cc) 断言:同样的 prompt,`GreedyGenerateCached` 与 `GreedyGenerate` **逐 token 完全一致**(`norm_topk` 两种取值都测)。

真机上,cached 与 no-cache 生成 id 也完全相同(→ `Paris.`);速度约 2× 于重算路径,且在 no-cache 会因大 T 崩溃的长度上稳跑。**又一次"新变体 == 老路径"的纪律。**

## 5. 一个模式的确立

到这一章,一个贯穿后续的模式已经清晰:**引擎每长出一个能带来风险的新变体(KV cache、导 trace、物理流式),就用一条 `== 已验证路径` 的 parity 单测把它锚住。** 这让 nuthatch 在不断加功能的同时,数值正确性从不失守。下一章的护城河、再下一章的物理流式,都遵循这条纪律。

下一章:[护城河:命中率、曲线、跨领域、跨架构](09-护城河)。
