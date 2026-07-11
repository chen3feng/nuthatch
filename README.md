# nuthatch

> **A GGUF MoE streaming inference engine in C++ — caches what matters, streams the rest.**

**nuthatch**(䴓)把种子塞进树皮缝里囤着、还记得位置;头朝下爬树。代号取自"囤食 + 记位置"——正是本项目的护城河:**usage-learned 逐专家流式缓存**。

## 这是什么

一个**支持 GGUF、跨 Mac/Linux/Windows、用 [blade-build](https://github.com/chen3feng/blade-build) 构建的 C++ MoE 流式推理引擎**。思想沿用 [colibrì](https://github.com/JustVugg/colibri):稠密权重常驻内存、路由专家留盘按需流式,让模型远大于内存也能跑。但 nuthatch **原生读 GGUF、原生 k-quant、多后端(CPU/Metal/CUDA/Vulkan via ggml)**。

护城河不是"引擎"(那是 ggml 已解决的 commodity),而是 **usage-learned 逐专家流式缓存 + 预取**这个算法,以及围绕它的可复现研究结论。

## 状态

早期开发中。按 [Master Roadmap (#1)](../../issues/1) 分步推进,每步一个 PR(带 workflow + UT)。

- `docs/ROADMAP.md` — 分步实现计划(M0→M5,PR#1–#18)
- `docs/dependencies.md` — 第三方依赖账本(引入前先讨论)
- `bench-log.md` — 研究日志(硬件实测数据点)

## 构建

```bash
blade test //...     # 需要 blade-build + ninja + vcpkg($VCPKG_ROOT)
```

## License

Apache 2.0(拟)。
