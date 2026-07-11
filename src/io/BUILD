# 通用的定位读字节层。tensor_source 本身不依赖 ggml(纯 pread),
# 与 GGUF/张量语义解耦——上层把 GgufReader 的 offset/size 传进来即可。
cc_library(
    name = 'tensor_source',
    srcs = ['tensor_source.cc'],
    hdrs = ['tensor_source.h'],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'tensor_source_test',
    srcs = ['tensor_source_test.cc'],
    deps = [
        ':tensor_source',
        '//src/gguf:gguf',           # 拿 data_offset + 张量 offset/size
        '//thirdparty/ggml:ggml',    # 测试自身造 GGUF fixture
    ],
)
