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

# 显存受限加载:非专家权重常驻,专家张量交给 ExpertReader 流式(真省内存)。
cc_library(
    name = 'streaming_model',
    srcs = ['streaming_model.cc'],
    hdrs = ['streaming_model.h'],
    deps = [
        ':expert_reader',
        ':olmoe_model',            # OlmoeConfig + ParseOlmoeConfig
        '//src/io:tensor_source',  # 跨平台定位读(常驻非专家权重)
        '//thirdparty/ggml:ggml',
    ],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'streaming_model_test',
    srcs = ['streaming_model_test.cc'],
    deps = [
        ':streaming_model',
        ':expert_reader',
        ':olmoe_model',
        '//thirdparty/ggml:ggml',
    ],
)

# 专家流式读取器:按 (张量名, expert_idx) 从 GGUF pread 出单个专家的量化字节。
cc_library(
    name = 'expert_reader',
    srcs = ['expert_reader.cc'],
    hdrs = ['expert_reader.h'],
    deps = [
        '//src/io:tensor_source',  # 跨平台定位读
        '//thirdparty/ggml:ggml',  # gguf 偏移/形状 + ggml 张量 nb/ne
    ],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'expert_reader_test',
    srcs = ['expert_reader_test.cc'],
    deps = [
        ':expert_reader',
        '//thirdparty/ggml:ggml',
    ],
)

# 有界专家槽缓存:每层 C 个槽,miss 从 reader 读入 gate/up/down,LRU 淘汰。
cc_library(
    name = 'expert_slot_cache',
    srcs = ['expert_slot_cache.cc'],
    hdrs = ['expert_slot_cache.h'],
    deps = [
        ':expert_reader',
        ':olmoe_model',            # OlmoeConfig
        '//thirdparty/ggml:ggml',
    ],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'expert_slot_cache_test',
    srcs = ['expert_slot_cache_test.cc'],
    deps = [
        ':expert_slot_cache',
        ':expert_reader',
        ':olmoe_model',
        '//thirdparty/ggml:ggml',
    ],
)

# 单序列 KV cache:每层常驻 K/V 缓存张量,跨 decode 步存活。
cc_library(
    name = 'kv_cache',
    srcs = ['kv_cache.cc'],
    hdrs = ['kv_cache.h'],
    deps = [
        ':olmoe_model',            # OlmoeConfig
        '//thirdparty/ggml:ggml',
    ],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'kv_cache_test',
    srcs = ['kv_cache_test.cc'],
    deps = [
        ':kv_cache',
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
        ':kv_cache',
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
        ':kv_cache',
        ':olmoe_model',
        '//src/trace:routing_trace',  # 导出真实推理路由 trace
        '//thirdparty/ggml:ggml',
    ],
    visibility = ['PUBLIC'],
)

# 物理流式前向:逐 token 两段式,专家按需装槽再算(真省内存)。
cc_library(
    name = 'streaming_forward',
    srcs = ['streaming_forward.cc'],
    hdrs = ['streaming_forward.h'],
    deps = [
        ':streaming_model',
        ':attention',
        ':expert_slot_cache',
        ':kv_cache',
        ':olmoe_model',
        '//thirdparty/ggml:ggml',
    ],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'streaming_forward_test',
    srcs = ['streaming_forward_test.cc'],
    deps = [
        ':streaming_forward',
        ':streaming_model',
        ':generate',               # GreedyGenerateCached 作 parity 基准
        ':olmoe_model',
        '//thirdparty/ggml:ggml',
    ],
)

cc_test(
    name = 'generate_test',
    srcs = ['generate_test.cc'],
    deps = [
        ':generate',
        ':olmoe_model',
        '//src/cache:cache_policy',   # Replay 真实 trace
        '//src/cache:lru_policy',
        '//src/trace:routing_trace',
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

# 物理流式推理 demo(CI 仅编译):专家按需装槽,输出文本 + 峰值内存。
cc_binary(
    name = 'olmoe_stream',
    srcs = ['olmoe_stream_main.cc'],
    deps = [
        ':streaming_forward',
        ':streaming_model',
        '//src/tokenizer:tokenizer',
    ],
)

# 护城河证据工具(CI 仅编译):真实推理导出路由 trace → 三策略比命中率。
cc_binary(
    name = 'olmoe_trace',
    srcs = ['olmoe_trace_main.cc'],
    deps = [
        ':generate',
        ':olmoe_model',
        '//src/tokenizer:tokenizer',
        '//src/trace:routing_trace',
        '//src/cache:cache_policy',
        '//src/cache:lru_policy',
        '//src/cache:os_page_cache_policy',
        '//src/cache:learned_pin_policy',
    ],
)
