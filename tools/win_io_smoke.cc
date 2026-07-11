// 可移植定位读(TensorSource)的功能冒烟——写临时文件 → ReadAt → 校验。
// Windows CI 用 MSVC(cl.exe)直接编它 + tensor_source.cc 并运行,验证
// ReadFile + OVERLAPPED 路径在真 Windows 上工作(不经 blade/ggml)。
// 不依赖 gtest,普通 main + 返回码。POSIX 上也能编能跑。
#include <cstdio>
#include <cstring>

#include "src/io/tensor_source.h"

int main() {
  const char* path = "win_io_smoke.tmp";
  const char data[] = "0123456789ABCDEF";  // 16 字节

  std::FILE* f = std::fopen(path, "wb");
  if (f == nullptr) return 1;
  std::fwrite(data, 1, 16, f);
  std::fclose(f);

  int rc = 0;
  {
    auto src = nuthatch::TensorSource::Open(path);
    if (src == nullptr) {
      rc = 2;
    } else {
      char buf[8] = {};
      // 从偏移 4 读 6 字节 → "456789"。
      const bool read_ok = src->ReadAt(4, 6, buf) &&
                           std::memcmp(buf, "456789", 6) == 0;
      // 越界读(偏移 10 读 50)应失败。
      char big[64];
      const bool oob_ok = !src->ReadAt(10, 50, big);
      if (!read_ok) rc = 3;
      else if (!oob_ok) rc = 4;
    }
  }

  std::remove(path);
  if (rc == 0) {
    std::printf("win_io_smoke OK (positional read + OOB detection)\n");
  } else {
    std::printf("win_io_smoke FAIL rc=%d\n", rc);
  }
  return rc;
}
