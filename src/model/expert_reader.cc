#include "src/model/expert_reader.h"

#include <fcntl.h>
#include <unistd.h>

#include <utility>

#include "ggml.h"
#include "gguf.h"

namespace nuthatch {

std::unique_ptr<ExpertReader> ExpertReader::Open(const std::string& gguf_path) {
  // no_alloc=true:只建元数据 + 张量形状(拿 ne/nb),不把权重读进内存。
  ggml_context* ctx = nullptr;
  gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/&ctx};
  gguf_context* g = gguf_init_from_file(gguf_path.c_str(), p);
  if (g == nullptr) return nullptr;

  const uint64_t data_off = gguf_get_data_offset(g);
  std::unordered_map<std::string, Info> info;
  for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr;
       t = ggml_get_next_tensor(ctx, t)) {
    const char* name = ggml_get_name(t);
    const int64_t id = gguf_find_tensor(g, name);
    if (id < 0) continue;
    Info in;
    in.file_offset = data_off + gguf_get_tensor_offset(g, id);
    in.expert_bytes = t->nb[2];  // dim2 一步的字节 = 一个专家的量化数据大小
    in.n_experts = static_cast<int>(t->ne[2]);
    info[name] = in;
  }
  gguf_free(g);
  ggml_free(ctx);

  const int fd = open(gguf_path.c_str(), O_RDONLY);
  if (fd < 0) return nullptr;
  return std::unique_ptr<ExpertReader>(new ExpertReader(fd, std::move(info)));
}

ExpertReader::ExpertReader(int fd, std::unordered_map<std::string, Info> info)
    : fd_(fd), info_(std::move(info)) {}

ExpertReader::~ExpertReader() {
  if (fd_ >= 0) close(fd_);
}

size_t ExpertReader::ExpertBytes(const std::string& name) const {
  auto it = info_.find(name);
  return it == info_.end() ? 0 : it->second.expert_bytes;
}

int ExpertReader::NumExperts(const std::string& name) const {
  auto it = info_.find(name);
  return it == info_.end() ? 0 : it->second.n_experts;
}

bool ExpertReader::ReadExpert(const std::string& name, int expert_idx,
                              void* out) const {
  auto it = info_.find(name);
  if (it == info_.end()) return false;
  const Info& in = it->second;
  if (expert_idx < 0 || expert_idx >= in.n_experts) return false;

  const off_t off = static_cast<off_t>(in.file_offset) +
                    static_cast<off_t>(expert_idx) *
                        static_cast<off_t>(in.expert_bytes);
  // pread:按绝对偏移读,不改文件位移,可多线程并发(未来多路预取)。
  const ssize_t n = pread(fd_, out, in.expert_bytes, off);
  return n == static_cast<ssize_t>(in.expert_bytes);
}

}  // namespace nuthatch
