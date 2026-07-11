#include "src/core/ggml_smoke.h"

#include "ggml.h"
#include "ggml-cpu.h"  // ggml_graph_compute_with_ctx

namespace nuthatch {

float GgmlMulSmoke(float a, float b) {
  // no_alloc=false:张量数据直接分配在 ctx 内,便于就地写输入、读输出。
  ggml_init_params params = {
      /*mem_size=*/ 16 * 1024 * 1024,
      /*mem_buffer=*/ nullptr,
      /*no_alloc=*/ false,
  };
  ggml_context* ctx = ggml_init(params);

  ggml_tensor* ta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  ggml_tensor* tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  ggml_tensor* tc = ggml_mul(ctx, ta, tb);  // 逐元素乘

  // 把 tc 标记为图的输出,展开出前向计算图。
  ggml_cgraph* graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, tc);

  // 图构建后再写输入(数据缓冲此时已分配)。
  static_cast<float*>(ta->data)[0] = a;
  static_cast<float*>(tb->data)[0] = b;

  ggml_graph_compute_with_ctx(ctx, graph, /*n_threads=*/1);

  const float result = static_cast<float*>(tc->data)[0];
  ggml_free(ctx);
  return result;
}

}  // namespace nuthatch
