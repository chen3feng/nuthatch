cc_library(
    name = 'gguf',
    srcs = ['gguf_reader.cc'],
    hdrs = ['gguf_reader.h'],
    deps = ['//thirdparty/ggml:ggml'],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'gguf_reader_test',
    srcs = ['gguf_reader_test.cc'],
    # 测试自身也直接调 ggml/gguf 写 API 造 fixture,故显式依赖 ggml。
    deps = [':gguf', '//thirdparty/ggml:ggml'],
)
