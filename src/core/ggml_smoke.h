#ifndef NUTHATCH_CORE_GGML_SMOKE_H_
#define NUTHATCH_CORE_GGML_SMOKE_H_

namespace nuthatch {

// 用 ggml 在 CPU 上计算 a*b 并返回结果。
// 仅用于冒烟:证明 blade 能链接 ggml、且 ggml 能跑通一次前向计算。
float GgmlMulSmoke(float a, float b);

}  // namespace nuthatch

#endif  // NUTHATCH_CORE_GGML_SMOKE_H_
