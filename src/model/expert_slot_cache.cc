#include "src/model/expert_slot_cache.h"

#include <utility>

namespace nuthatch {

std::unique_ptr<ExpertSlotCache> ExpertSlotCache::Create(const OlmoeConfig& cfg,
                                                         int layer, int capacity,
                                                         ExpertReader* reader) {
  if (capacity <= 0 || reader == nullptr) return nullptr;
  const std::string pfx = "blk." + std::to_string(layer) + ".";
  const std::string gate = pfx + "ffn_gate_exps.weight";
  const std::string up = pfx + "ffn_up_exps.weight";
  const std::string down = pfx + "ffn_down_exps.weight";

  const int32_t gt = reader->ExpertType(gate);
  const int32_t ut = reader->ExpertType(up);
  const int32_t dt = reader->ExpertType(down);
  if (gt < 0 || ut < 0 || dt < 0) return nullptr;  // 缺专家张量

  // 槽张量按真专家的类型/形状分配,第三维 = 容量。量化类型下 nb[2] 与真张量
  // 一致,故 reader 读出的字节能原样填进槽。
  const size_t bytes = (reader->ExpertBytes(gate) + reader->ExpertBytes(up) +
                        reader->ExpertBytes(down)) *
                       static_cast<size_t>(capacity);
  const size_t mem = bytes + 3 * ggml_tensor_overhead() + (1u << 20);
  ggml_init_params ip = {mem, nullptr, /*no_alloc=*/false};
  ggml_context* ctx = ggml_init(ip);

  auto self = std::unique_ptr<ExpertSlotCache>(
      new ExpertSlotCache(reader, ctx, capacity, gate, up, down));
  const int64_t n_embd = cfg.n_embd;
  const int64_t n_ff = cfg.n_ff;
  self->gate_ = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(gt), n_embd, n_ff,
                                   capacity);
  self->up_ = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ut), n_embd, n_ff,
                                 capacity);
  self->down_ = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(dt), n_ff, n_embd,
                                   capacity);
  return self;
}

ExpertSlotCache::ExpertSlotCache(ExpertReader* reader, ggml_context* ctx,
                                 int capacity, std::string gate, std::string up,
                                 std::string down)
    : reader_(reader),
      ctx_(ctx),
      gate_name_(std::move(gate)),
      up_name_(std::move(up)),
      down_name_(std::move(down)),
      capacity_(capacity),
      slot_to_id_(capacity, -1),
      slot_tick_(capacity, 0) {}

ExpertSlotCache::~ExpertSlotCache() {
  if (ctx_ != nullptr) ggml_free(ctx_);
}

int ExpertSlotCache::AcquireSlot() {
  for (int s = 0; s < capacity_; ++s) {
    if (slot_to_id_[s] < 0) return s;  // 空槽优先
  }
  // 否则淘汰 tick 最小(最久未用)的槽。因 distinct(ids)≤capacity 且本批已装入
  // 的专家 tick 最大,淘汰的必是更早批次的专家,不会误踢本批刚装的。
  int victim = 0;
  for (int s = 1; s < capacity_; ++s) {
    if (slot_tick_[s] < slot_tick_[victim]) victim = s;
  }
  id_to_slot_.erase(slot_to_id_[victim]);
  slot_to_id_[victim] = -1;
  return victim;
}

bool ExpertSlotCache::LoadSlot(int slot, int id) {
  char* g = static_cast<char*>(gate_->data) + slot * gate_->nb[2];
  char* u = static_cast<char*>(up_->data) + slot * up_->nb[2];
  char* d = static_cast<char*>(down_->data) + slot * down_->nb[2];
  return reader_->ReadExpert(gate_name_, id, g) &&
         reader_->ReadExpert(up_name_, id, u) &&
         reader_->ReadExpert(down_name_, id, d);
}

std::unordered_map<int, int> ExpertSlotCache::Ensure(
    const std::vector<int>& expert_ids) {
  std::unordered_map<int, int> result;
  for (int id : expert_ids) {
    if (result.count(id)) continue;  // 同批去重
    auto it = id_to_slot_.find(id);
    int slot;
    if (it != id_to_slot_.end()) {
      slot = it->second;  // 命中
      ++hits_;
    } else {
      ++misses_;  // 缺失:占槽 + 从盘读入
      slot = AcquireSlot();
      LoadSlot(slot, id);
      slot_to_id_[slot] = id;
      id_to_slot_[id] = slot;
    }
    slot_tick_[slot] = ++tick_;  // 刷新最近使用
    result[id] = slot;
  }
  return result;
}

}  // namespace nuthatch
