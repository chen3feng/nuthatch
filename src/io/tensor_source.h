#ifndef NUTHATCH_IO_TENSOR_SOURCE_H_
#define NUTHATCH_IO_TENSOR_SOURCE_H_

#include <cstddef>
#include <memory>
#include <string>

namespace nuthatch {

// 从一个文件按【绝对字节偏移】定位读取原始字节,底层用 pread。
//
// 为什么不 mmap:mmap 会让读过的页留在进程 RSS 里、由内核按需回收,难以对
// "模型远大于内存"的专家流式做精确的驻留控制。pread + 后续的 posix_fadvise
// (DONTNEED)才能读完即弃、把 RSS 压住。这是整个流式加载的地基。
//
// 线程安全:pread 带偏移、不改动 fd 的读写位置,多线程可并发读同一实例。
class TensorSource {
 public:
  // 以只读方式打开文件;失败返回 nullptr。
  static std::unique_ptr<TensorSource> Open(const std::string& path);
  ~TensorSource();

  TensorSource(const TensorSource&) = delete;
  TensorSource& operator=(const TensorSource&) = delete;

  // 从 offset 起读 size 字节到 out。完整读满返回 true;
  // 读错误或提前 EOF(越界)返回 false。
  bool ReadAt(size_t offset, size_t size, void* out) const;

 private:
  explicit TensorSource(int fd) : fd_(fd) {}

  int fd_ = -1;
};

}  // namespace nuthatch

#endif  // NUTHATCH_IO_TENSOR_SOURCE_H_
