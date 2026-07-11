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
