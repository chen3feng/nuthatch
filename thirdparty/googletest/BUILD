# vcpkg 提供的 googletest 的薄封装(照 flare)。
# consumers 用 //thirdparty/googletest:gtest;cc_test 会自动链 gtest_main
# (见 BLADE_ROOT 的 cc_test_config),一般无需手动依赖。
# 注:blade 要求 cc_library 显式声明 hdrs——这里的头文件由 vcpkg 端口提供,
# 本地无公开头,故 hdrs = []。

cc_library(
    name = 'gtest',
    hdrs = [],
    deps = ['vcpkg#gtest:gtest'],
    visibility = ['PUBLIC'],
)

cc_library(
    name = 'gtest_main',
    hdrs = [],
    deps = ['vcpkg#gtest:gtest_main', ':gtest'],
    visibility = ['PUBLIC'],
)

cc_library(
    name = 'gmock',
    hdrs = [],
    deps = ['vcpkg#gtest:gmock', ':gtest'],
    visibility = ['PUBLIC'],
)

cc_library(
    name = 'gmock_main',
    hdrs = [],
    deps = ['vcpkg#gtest:gmock_main', ':gmock'],
    visibility = ['PUBLIC'],
)
