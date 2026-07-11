# GPT-2 byte-level BPE 分词器。解码已在 P16;编码器(PCRE2 预分词 + BPE)本 PR。
cc_library(
    name = 'tokenizer',
    srcs = ['tokenizer.cc'],
    hdrs = ['tokenizer.h'],
    deps = [
        '//thirdparty/ggml:ggml',    # 读 GGUF 的 tokenizer.ggml.tokens/merges
        '//thirdparty/pcre2:pcre2',  # 预分词的 Unicode 正则
    ],
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

cc_test(
    name = 'pcre2_smoke_test',
    srcs = ['pcre2_smoke_test.cc'],
    deps = ['//thirdparty/pcre2:pcre2'],
)
