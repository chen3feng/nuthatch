# GPT-2 byte-level BPE 分词器。本 PR 只有解码(id→文本);编码器后续 PR。
cc_library(
    name = 'tokenizer',
    srcs = ['tokenizer.cc'],
    hdrs = ['tokenizer.h'],
    deps = ['//thirdparty/ggml:ggml'],  # 读 GGUF 的 tokenizer.ggml.tokens
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'tokenizer_test',
    srcs = ['tokenizer_test.cc'],
    deps = [
        ':tokenizer',
        '//thirdparty/ggml:ggml',
    ],
)
