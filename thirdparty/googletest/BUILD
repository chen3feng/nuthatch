# vcpkg 提供的 googletest 的薄封装(照 flare)。
# consumers 用 //thirdparty/googletest:gtest;cc_test 会自动链 gtest_main
# (见 BLADE_ROOT 的 cc_test_config),一般无需手动依赖。

cc_library(
    name = 'gtest',
    deps = ['vcpkg#gtest:gtest'],
    visibility = ['PUBLIC'],
)

cc_library(
    name = 'gtest_main',
    deps = ['vcpkg#gtest:gtest_main', ':gtest'],
    visibility = ['PUBLIC'],
)

cc_library(
    name = 'gmock',
    deps = ['vcpkg#gtest:gmock', ':gtest'],
    visibility = ['PUBLIC'],
)

cc_library(
    name = 'gmock_main',
    deps = ['vcpkg#gtest:gmock_main', ':gmock'],
    visibility = ['PUBLIC'],
)
