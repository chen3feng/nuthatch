# 路由 trace 的格式与读写。纯 std,无第三方依赖——缓存模拟器(P8+)的输入层。
cc_library(
    name = 'routing_trace',
    srcs = ['routing_trace.cc'],
    hdrs = ['routing_trace.h'],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'routing_trace_test',
    srcs = ['routing_trace_test.cc'],
    deps = [':routing_trace'],
)
