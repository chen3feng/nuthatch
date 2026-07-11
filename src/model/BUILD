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
