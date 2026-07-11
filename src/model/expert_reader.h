#ifndef NUTHATCH_MODEL_EXPERT_READER_H_
#define NUTHATCH_MODEL_EXPERT_READER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace nuthatch {

// 从 GGUF 文件按需读取【单个专家】的量化权重字节(不常驻整张专家张量)。
// 专家张量是 3D [ne0, ne1, n_expert];第 e 个专家 = 沿 dim2 的一个切片,
// 字节大小 = nb[2],文件位置 = data_offset + tensor_offset + e*nb[2]。
// 这是流式 MoE(P20)的地基:miss 时从盘 pread 出被选中的专家、填进槽缓存。
//
// I/O:POSIX pread(线程安全、不动文件位移)。Windows 的 ReadFile/OVERLAPPED
// 路径是后续 TODO(colibrì 亦未做)。
class ExpertReader {
 public:
  // 打开 GGUF(只读元数据拿偏移/步长,不载权重数据)并保持文件 fd。失败返回 nullptr。
  static std::unique_ptr<ExpertReader> Open(const std::string& gguf_path);
  ~ExpertReader();

  ExpertReader(const ExpertReader&) = delete;
  ExpertReader& operator=(const ExpertReader&) = delete;

  // 一个专家占的字节数(= nb[2]);张量不存在返回 0。
  size_t ExpertBytes(const std::string& tensor_name) const;
  // 专家数(= ne[2]);张量不存在返回 0。
  int NumExperts(const std::string& tensor_name) const;
  // 元素类型(ggml_type,as int;量化如 Q4_K);张量不存在返回 -1。
  // 建槽缓存张量时要按同类型分配,量化字节才能直接喂 mul_mat。
  int32_t ExpertType(const std::string& tensor_name) const;

  // 把 tensor_name 的第 expert_idx 个专家的量化字节读进 out(须 ≥ ExpertBytes)。
  // 成功返回 true;out 内容可直接作为 ggml 量化数据喂 mul_mat。
  bool ReadExpert(const std::string& tensor_name, int expert_idx,
                  void* out) const;

 private:
  struct Info {
    uint64_t file_offset;  // 文件内该张量数据起点(data_offset + tensor_offset)
    size_t expert_bytes;   // 一个专家的字节数(nb[2])
    int n_experts;         // ne[2]
    int32_t type;          // 元素类型(ggml_type as int)
  };
  ExpertReader(int fd, std::unordered_map<std::string, Info> info);

  int fd_ = -1;
  std::unordered_map<std::string, Info> info_;
};

}  // namespace nuthatch

#endif  // NUTHATCH_MODEL_EXPERT_READER_H_
