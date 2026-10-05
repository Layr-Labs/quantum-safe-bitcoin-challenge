// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstring>

namespace qsb {
// Input storage is read-only to consumers. Before enqueue(slot), the caller
// must complete that slot's prior work (including any copy that read its input).
// The original allocations remain owned by the caller; only events are owned here.
template<int Slots> class SharedInputPool {
 public:
  SharedInputPool() { for (int s=0;s<Slots;s++) binding_[s]=-1; }
  ~SharedInputPool() {
    for (int i=0;i<Slots;i++) if (entries_[i].ready) cudaEventDestroy(entries_[i].ready);
  }
  SharedInputPool(const SharedInputPool&)=delete;
  SharedInputPool& operator=(const SharedInputPool&)=delete;

  cudaError_t init(int index, void* device, void* pinned_host, size_t bytes) {
    if(index<0 || index>=Slots || !device || !pinned_host || !bytes || entries_[index].device)
      return cudaErrorInvalidValue;
    if(bytes_ && bytes_!=bytes) return cudaErrorInvalidValue;
    cudaEvent_t event=nullptr;
    cudaError_t e=cudaEventCreateWithFlags(&event,cudaEventDisableTiming);
    if(e!=cudaSuccess) return e;
    entries_[index].device=device;
    entries_[index].host=pinned_host;
    entries_[index].ready=event;
    bytes_=bytes; initialized_++;
    return cudaSuccess;
  }

  cudaError_t enqueue(int slot, const void* input, size_t bytes, cudaStream_t stream,
                      const void** device_out, bool* uploaded=nullptr) {
    if(poisoned_ || slot<0 || slot>=Slots || initialized_!=Slots || !input || !device_out || bytes!=bytes_)
      return cudaErrorInvalidValue;
    if(uploaded) *uploaded=false;
    // The caller completed this slot, so its prior reference is now releasable.
    int old=binding_[slot];
    if(old>=0) { entries_[old].refs--; binding_[slot]=-1; }
    int selected=-1;
    for(int i=0;i<Slots;i++)
      if(entries_[i].valid && std::memcmp(entries_[i].host,input,bytes_)==0) { selected=i; break; }
    if(selected>=0) {
      // Another stream may still be uploading this shared record. Even cache
      // hits must join that transfer before their consumers read the bytes.
      cudaError_t e=cudaStreamWaitEvent(stream,entries_[selected].ready,0);
      if(e!=cudaSuccess) { poisoned_=true; return e; }
    } else {
      // At most Slots-1 consumers remain after releasing this slot, so one
      // entry is free. Never overwrite a record still referenced by a consumer.
      for(int i=0;i<Slots;i++) if(!entries_[i].refs) { selected=i; break; }
      if(selected<0) return cudaErrorInvalidValue;
      Entry& entry=entries_[selected];
      entry.valid=false;
      std::memcpy(entry.host,input,bytes_);
      cudaError_t e=cudaMemcpyAsync(entry.device,entry.host,bytes_,cudaMemcpyHostToDevice,stream);
      if(e==cudaSuccess) e=cudaEventRecord(entry.ready,stream);
      if(e!=cudaSuccess) { poisoned_=true; return e; }
      entry.valid=true;
      if(uploaded) *uploaded=true;
    }
    entries_[selected].refs++;
    binding_[slot]=selected;
    *device_out=entries_[selected].device;
    return cudaSuccess;
  }
 private:
  struct Entry { void* device=nullptr; void* host=nullptr; cudaEvent_t ready=nullptr; int refs=0; bool valid=false; };
  Entry entries_[Slots];
  int binding_[Slots];
  int initialized_=0;
  size_t bytes_=0;
  bool poisoned_=false;
};
} // namespace qsb
