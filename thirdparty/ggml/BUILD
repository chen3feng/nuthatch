# vcpkg 提供的 ggml 的薄封装。consumers 用 //thirdparty/ggml:ggml。
# hdrs = []:公开头(ggml.h / ggml-cpu.h / ggml-backend.h)由 vcpkg 端口提供。
#
# ggml 的符号分散在三个静态库(nm 实测):
#   libggml.a       空壳 umbrella(无符号)
#   libggml-base.a  核心算子(ggml_init / ggml_mul / ggml_new_tensor_1d /
#                             ggml_new_graph / ggml_build_forward_expand / ggml_free …)
#   libggml-cpu.a   CPU backend(ggml_graph_compute_with_ctx)
# blade 的 vcpkg 集成只链 umbrella、不跟随 CMake 的 transitive link interface,
# 故必须把 base / cpu 显式列出(见 docs/blade-feedback.md)。
# 顺序:依赖者在前(cpu -> base),static 链接时符号解析更稳。

cc_library(
    name = 'ggml',
    hdrs = [],
    deps = [
        'vcpkg#ggml:ggml-cpu',
        'vcpkg#ggml:ggml-base',
        'vcpkg#ggml:ggml',
    ],
    visibility = ['PUBLIC'],
)
