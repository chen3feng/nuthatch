# vcpkg 提供的 ggml 的薄封装。consumers 用 //thirdparty/ggml:ggml。
# hdrs = []:公开头(ggml.h / ggml-cpu.h / ggml-backend.h)由 vcpkg 端口提供,
# 本地无公开头。CPU backend(ggml-cpu)随 ggml 端口一并链入。
# metal/cuda/vulkan feature 到 M4(PR#15)按平台开启。

cc_library(
    name = 'ggml',
    hdrs = [],
    deps = ['vcpkg#ggml:ggml'],
    visibility = ['PUBLIC'],
)
