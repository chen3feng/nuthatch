# 缓存策略模拟器:统一接口 + 重放器 + 具体策略。研究"隔离变量对比策略"的核心。
cc_library(
    name = 'cache_policy',
    srcs = ['cache_policy.cc'],
    hdrs = ['cache_policy.h'],
    deps = ['//src/trace:routing_trace'],
    visibility = ['PUBLIC'],
)

cc_library(
    name = 'lru_policy',
    srcs = ['lru_policy.cc'],
    hdrs = ['lru_policy.h'],
    deps = [':cache_policy'],
    visibility = ['PUBLIC'],
)

cc_test(
    name = 'lru_policy_test',
    srcs = ['lru_policy_test.cc'],
    deps = [
        ':lru_policy',
        ':cache_policy',              # 直接用 Replay/ReplayStats
        '//src/trace:routing_trace',  # 直接构造 RoutingTrace
    ],
)
