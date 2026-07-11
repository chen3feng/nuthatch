#ifndef NUTHATCH_TRACE_ROUTING_TRACE_H_
#define NUTHATCH_TRACE_ROUTING_TRACE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace nuthatch {

// 一次 (token, layer) 的路由决策:该 MoE 层为该 token 选中的专家 id。
struct RoutingRecord {
  uint32_t token;                 // 第几个生成的 token(按顺序)
  uint32_t layer;                 // MoE 层号
  std::vector<uint32_t> experts;  // 选中的专家 id(top-k,可变长)
};

// 一整段 decode 的路由轨迹。缓存策略模拟器(P8+)重放它来算命中率/读字节,
// 从而在【不跑完整引擎、几乎不占盘】的前提下对比不同缓存策略。
struct RoutingTrace {
  uint32_t n_layers = 0;
  uint32_t n_expert = 0;  // 每层专家总数
  std::vector<RoutingRecord> records;
};

// 紧凑二进制格式。注意:用本机字节序,仅作【本地研究工装】,不是跨机 wire 格式。
// 写:成功返回 true。
bool WriteRoutingTrace(const RoutingTrace& trace, const std::string& path);
// 读:成功填充 *out 并返回 true;文件缺失/魔数错/版本错/截断/损坏返回 false。
bool ReadRoutingTrace(const std::string& path, RoutingTrace* out);

// 读【文本】trace:每行 "<layer> e0 e1 ...",由外部工具(如打了补丁的 llama.cpp
// eval-callback)导出,用于跨架构验证缓存策略。n_layers/n_expert 由内容推断
// (max+1)。成功(至少一条记录)返回 true。
bool ReadRoutingTraceText(const std::string& path, RoutingTrace* out);

}  // namespace nuthatch

#endif  // NUTHATCH_TRACE_ROUTING_TRACE_H_
