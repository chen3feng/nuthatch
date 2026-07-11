cc_library(
    name = 'ggml_smoke',
    srcs = ['ggml_smoke.cc'],
    hdrs = ['ggml_smoke.h'],
    deps = ['//thirdparty/ggml:ggml'],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'ggml_smoke_test',
    srcs = ['ggml_smoke_test.cc'],
    deps = [':ggml_smoke'],   # gtest_main 由 BLADE_ROOT cc_test_config 自动注入
)
