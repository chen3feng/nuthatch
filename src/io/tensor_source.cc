#include "src/io/tensor_source.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>

namespace nuthatch {

std::unique_ptr<TensorSource> TensorSource::Open(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return nullptr;
  return std::unique_ptr<TensorSource>(new TensorSource(fd));
}

TensorSource::~TensorSource() {
  if (fd_ >= 0) ::close(fd_);
}

bool TensorSource::ReadAt(size_t offset, size_t size, void* out) const {
  auto* dst = static_cast<uint8_t*>(out);
  size_t done = 0;
  // pread 允许一次只返回部分数据(短读),循环补齐直到读满或出错/EOF。
  while (done < size) {
    const ssize_t n = ::pread(fd_, dst + done, size - done, offset + done);
    if (n < 0) return false;   // 读错误
    if (n == 0) return false;  // 提前 EOF —— 请求越界
    done += static_cast<size_t>(n);
  }
  return true;
}

}  // namespace nuthatch
