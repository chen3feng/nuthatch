# OLMoE 模型:config + 权重加载(先常驻;流式后续接入)。走向"能推理"的地基。
cc_library(
    name = 'olmoe_model',
    srcs = ['olmoe_model.cc'],
    hdrs = ['olmoe_model.h'],
    deps = ['//thirdparty/ggml:ggml'],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'olmoe_model_test',
    srcs = ['olmoe_model_test.cc'],
    deps = [
        ':olmoe_model',
        '//thirdparty/ggml:ggml',  # 测试自造 tiny-OLMoE GGUF fixture
    ],
)

cc_library(
    name = 'attention',
    srcs = ['attention.cc'],
    hdrs = ['attention.h'],
    deps = [
        ':olmoe_model',            # OlmoeConfig
        '//thirdparty/ggml:ggml',
    ],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'attention_test',
    srcs = ['attention_test.cc'],
    deps = [
        ':attention',
        ':olmoe_model',
        '//thirdparty/ggml:ggml',
    ],
)

cc_library(
    name = 'moe',
    srcs = ['moe.cc'],
    hdrs = ['moe.h'],
    deps = [
        ':olmoe_model',
        '//thirdparty/ggml:ggml',
    ],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'moe_test',
    srcs = ['moe_test.cc'],
    deps = [
        ':moe',
        ':olmoe_model',
        '//thirdparty/ggml:ggml',
    ],
)

cc_library(
    name = 'forward',
    srcs = ['forward.cc'],
    hdrs = ['forward.h'],
    deps = [
        ':olmoe_model',
        ':attention',
        ':moe',
        '//thirdparty/ggml:ggml',
    ],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'forward_test',
    srcs = ['forward_test.cc'],
    deps = [
        ':forward',
        ':olmoe_model',
        '//thirdparty/ggml:ggml',
    ],
)

# 本地调试:在真 OLMoE 上跑前向看 logits(CI 不跑,仅编译)。
cc_binary(
    name = 'olmoe_forward',
    srcs = ['olmoe_forward_main.cc'],
    deps = [
        ':forward',
        ':olmoe_model',
        '//thirdparty/ggml:ggml',
    ],
)

cc_library(
    name = 'generate',
    srcs = ['generate.cc'],
    hdrs = ['generate.h'],
    deps = [
        ':forward',
        ':olmoe_model',
        '//thirdparty/ggml:ggml',
    ],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'generate_test',
    srcs = ['generate_test.cc'],
    deps = [
        ':generate',
        ':olmoe_model',
        '//thirdparty/ggml:ggml',
    ],
)

# 本地推理 + 对拍工具(CI 仅编译)。
cc_binary(
    name = 'olmoe_generate',
    srcs = ['olmoe_generate_main.cc'],
    deps = [
        ':generate',
        ':olmoe_model',
        '//src/tokenizer:tokenizer',  # 解码生成文本
    ],
)
