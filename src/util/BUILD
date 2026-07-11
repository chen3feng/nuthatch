cc_library(
    name = 'version',
    srcs = ['version.cc'],
    hdrs = ['version.h'],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'version_test',
    srcs = ['version_test.cc'],
    deps = [':version'],   # gtest_main 由 BLADE_ROOT cc_test_config 自动注入
)
