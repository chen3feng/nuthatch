#include "src/io/tensor_source.h"

#include <cstdint>

#ifdef _WIN32
#include <windows.h>

#include <algorithm>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace nuthatch {

#ifdef _WIN32

std::unique_ptr<TensorSource> TensorSource::Open(const std::string& path) {
  HANDLE h = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return nullptr;
  return std::unique_ptr<TensorSource>(new TensorSource(static_cast<void*>(h)));
}

TensorSource::~TensorSource() {
  if (handle_ != nullptr) ::CloseHandle(static_cast<HANDLE>(handle_));
}

bool TensorSource::ReadAt(size_t offset, size_t size, void* out) const {
  auto* dst = static_cast<uint8_t*>(out);
  size_t done = 0;
  // OVERLAPPED 带偏移做定位读;ReadFile 一次最多 DWORD 字节,循环补齐/兜短读。
  while (done < size) {
    const uint64_t off = static_cast<uint64_t>(offset) + done;
    OVERLAPPED ov = {};
    ov.Offset = static_cast<DWORD>(off & 0xFFFFFFFFu);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    const DWORD want =
        static_cast<DWORD>(std::min<size_t>(size - done, 0xFFFFFFFFu));
    DWORD got = 0;
    if (!::ReadFile(static_cast<HANDLE>(handle_), dst + done, want, &got, &ov)) {
      return false;  // 读错误
    }
    if (got == 0) return false;  // 提前 EOF —— 请求越界
    done += got;
  }
  return true;
}

#else  // POSIX

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

#endif

}  // namespace nuthatch
