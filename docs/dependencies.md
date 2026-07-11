# 依赖账本 (dependencies ledger)

> 规则:任何第三方库(C++ 或 Python)引入前**先讨论、达成一致**,再写进 BUILD / requirements。
> 保持最小。状态:`提议` / `已同意` / `已引入` / `推迟` / `否决`。

## C++ 运行时依赖(每个都要慎重 vet)

| 库 | 用途 | 许可 | 哪个 PR | 状态 | 备注 |
|---|---|---|---|---|---|
| **ggml** | 张量 + k-quant 内核 + 多后端(CPU/Metal/CUDA/Vulkan)+ GGUF 读取(`gguf.h`) | MIT | #2 起 | **✅ 已同意** | 走 blade vcpkg:`vcpkg#ggml:ggml`(薄封装 `//thirdparty/ggml`),vcpkg pin v0.11.1。若日后需要更新架构支持再议切本地源码 |
| **googletest** | 单测框架 | BSD-3 | #1 起 | **✅ 已同意** | 走 blade vcpkg:`vcpkg#gtest`(照 flare,pin 版本);`cc_test` 自动注入 gtest_main,不显式写 |
| cpp-httplib | 可选 C++ HTTP server(单头、SSE) | MIT | #18(M5,可选) | **推迟** | 只有走"C++ 单体二进制"才需要;否则 HTTP 走 Python 网关 |
| JSON 库(nlohmann/json?) | 若 C++ 侧要读/写 JSON | MIT | 待定 | **尽量避免** | GGUF 元数据走 ggml 的 gguf API,不需 JSON;配置尽量简单/自写。除非 M5 C++ server 需要 |

## Python 工具依赖(`tools/`,不进运行时)

| 库 | 用途 | 许可 | 哪个 PR | 状态 |
|---|---|---|---|---|
| numpy | 数值/trace 分析 | BSD | #10 起 | 提议 |
| matplotlib | 画命中率曲线等研究图 | PSF/BSD | #10 起 | 提议 |
| huggingface_hub | 下载模型/fixture(也可直接用 llama.cpp `-hf` 或手动) | Apache-2 | #7/#13 | 提议(可选) |
| gguf | 离线探查 GGUF 元数据/张量(装 OLMoE 架构时用) | MIT | 工具 | ✅ 已同意(用户批准) |

## 系统/平台(随 ggml 构建,非独立库,不单独 vet)

- Metal framework(macOS)、CUDA Toolkit、Vulkan SDK —— 由对应 ggml backend 带入,平台条件编译(PR#15+)。

## 决策记录

- 2026-07-11 立项:确立"依赖引入前先讨论"规则;上表为路线图**预期**依赖,尚未逐个拍板。
- 待拍板(启动 #1/#2 需要):**googletest**、**ggml**。
