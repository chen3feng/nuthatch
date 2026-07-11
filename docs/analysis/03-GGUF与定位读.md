---
title: 读 GGUF 与定位读
nav_order: 4
---

# 读 GGUF 与定位读:pread 而非 mmap

## 1. 为什么押注 GGUF

第一个格式决定:**吃 GGUF,不自造格式。** GGUF 是本地大模型推理的事实标准(llama.cpp / ggml 生态)。押它等于**站在生态上**:

- HuggingFace 上现成的量化模型随手可得,`ollama pull` 也直接下——不用自己转换;
- ggml **原生读 GGUF、原生 k-quant**,不必重写量化内核;
- 工具链成熟,`llama-tokenize` / `llama-cli` 正好当**对拍基准**(第 6、7 章)。

colibrì 自造格式省了依赖却也断了生态。nuthatch 反过来,押注普及度与库支持——代价是要理解 GGUF 的张量布局,而这恰好是流式的地基。

## 2. GGUF 张量布局与专家的"切片"

GGUF 文件 = 元数据(超参 KV + 每个张量的名字/类型/形状/偏移)+ 紧接着的张量数据区。ggml 的 `gguf_init_from_file` 用 `no_alloc=true` 可以**只读元数据、不载数据**,这对流式至关重要。

MoE 的专家权重是**融合的 3D 张量** `[n_embd, n_ff, n_expert]`——所有专家挤在一个张量里。第 `e` 个专家 = 沿第三维的一个切片:

- 字节大小 = `nb[2]`(ggml 按量化类型算好的 dim2 步长);
- 文件位置 = `gguf_get_data_offset + gguf_get_tensor_offset + e * nb[2]`。

这个观察是 [`expert_reader.cc`](../../src/model/expert_reader.cc) 的全部依据:要流式读单个专家,不必载整张张量,只要 `pread` 出那一段字节。而且是**量化字节直接读、直接喂 `mul_mat`**,不需要反量化。

## 3. 核心决定:pread,而非 mmap

整个项目最早、也最关键的 I/O 判断:**用 `pread` 定位读,不用 `mmap`。**

理由直指护城河。流式缓存要**可控地预取和驱逐**专家——决定谁留内存、谁被踢、什么时候预读。`mmap` 把这些全交给操作系统换页:你读过的页留在 RSS 里,由内核按它自己的 LRU 回收。**而"操作系统页缓存"正是我们想打败的基线**——它就实现在 [`cache/os_page_cache_policy`](../../src/cache/os_page_cache_policy.h),作为对照组。

如果用 `mmap`,就等于把缓存策略拱手让给内核,学习策略无从谈起。只有自己 `pread`,才谈得上"用历史使用决定驻留"。这是"要打败 X,就不能把控制权交给 X"的一个具体例子。

## 4. TensorSource:一次读、跨平台

[`src/io/tensor_source`](../../src/io/tensor_source.cc) 把定位读收敛成一个接口:

```cpp
class TensorSource {
  static std::unique_ptr<TensorSource> Open(const std::string& path);
  // 从 offset 起读 size 字节到 out;读满 true,出错/提前 EOF false。
  bool ReadAt(size_t offset, size_t size, void* out) const;
};
```

`ReadAt` 是**带偏移、不改文件位置、可并发**的定位读(为将来多路预取铺路)。它循环补齐短读。

跨平台(第 11 章的 Windows 工作在此落地):

```cpp
#ifdef _WIN32   // ReadFile + OVERLAPPED —— 同样是带偏移、不改位置的定位读
#else           // POSIX pread
#endif
```

`ExpertReader`(读专家)和 `StreamingModel`(读常驻的非专家权重,第 10 章)都复用它——一处可移植、DRY 掉三份裸 `pread`。

## 5. 分层地基

至此,`src/gguf`(读元数据)+ `src/io`(读字节)+ `src/moe`(专家张量布局)构成了引擎最底层的地基。它们不知道 OLMoE、不知道注意力,只知道"怎么从一个 GGUF 文件里,按名字/偏移,读出(或不读出)张量的字节"。上层的引擎(第 5 章)与护城河(第 9、10 章)都长在这块地基上。

下一章:[研究先行:trace 与缓存策略](04-研究先行)。
