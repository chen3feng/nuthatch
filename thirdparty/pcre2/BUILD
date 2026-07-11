# vcpkg 提供的 PCRE2(8-bit)的薄封装。consumers 用 //thirdparty/pcre2:pcre2。
# 选 PCRE2 而非 RE2:同时支持 \p{L}/\p{N} 和前瞻 (?!\S),能跑完整 OLMo 正则;
# 且 self-contained,不像新版 RE2 那样拖 abseil(→macOS CoreFoundation 链接坑)。
cc_library(
    name = 'pcre2',
    hdrs = [],
    deps = ['vcpkg#pcre2:pcre2-8'],
    visibility = ['PUBLIC'],
)
