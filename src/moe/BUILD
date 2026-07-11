cc_library(
    name = 'expert_layout',
    srcs = ['expert_layout.cc'],
    hdrs = ['expert_layout.h'],
    deps = [
        '//src/gguf:gguf',
        '//thirdparty/ggml:ggml',  # 头里用到 ggml_type
    ],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'expert_layout_test',
    srcs = ['expert_layout_test.cc'],
    deps = [
        ':expert_layout',
        '//src/gguf:gguf',           # 直接调 GgufReader::Open(须显式声明)
        '//src/io:tensor_source',    # 按切片读回验证
        '//thirdparty/ggml:ggml',    # 测试自身造 GGUF fixture
    ],
)
