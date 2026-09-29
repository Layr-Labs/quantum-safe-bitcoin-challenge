#!/usr/bin/env python3
"""Compile the actual host helper against a deferred CUDA-copy mock.

Checks layout, the unchanged hit prefix, stream/event ordering, slot reuse and
error propagation. This executes no GPU code and makes no performance claim.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CUDA_MOCK = r"""
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
typedef int cudaError_t;
const int cudaSuccess=0, cudaErrorInvalidValue=1, injectedError=73;
const unsigned cudaHostAllocDefault=0;
const int cudaMemcpyDeviceToHost=2;
struct MockStream {
  std::vector<std::function<void()> > work;
  size_t executed=0;
};
struct MockEvent {
  MockStream* stream=nullptr;
  size_t fence=0;
  unsigned generation=0;
};
typedef MockStream* cudaStream_t;
typedef MockEvent* cudaEvent_t;
struct Allocation { void* ptr; size_t size; bool host,alive; };
static std::vector<Allocation> allocations;
static std::vector<std::string> calls;
static int failAt=0;
static size_t copyCount=0, recordCount=0;
static void require(bool ok,const char* why) {
  if (!ok) throw std::runtime_error(why);
}
static cudaError_t api(const char* name) {
  calls.push_back(name);
  return static_cast<int>(calls.size())==failAt ? injectedError : cudaSuccess;
}
static Allocation& allocation(const void* ptr,bool host) {
  for (Allocation& a:allocations)
    if (a.ptr==ptr && a.alive && a.host==host) return a;
  throw std::runtime_error("unknown allocation or wrong allocation kind");
}
static cudaError_t allocate(void** ptr,size_t size,bool host) {
  cudaError_t e=api(host ? "host-alloc" : "device-alloc");
  if (e) return e;
  void* p=std::malloc(size);
  require(p!=nullptr,"mock malloc failed");
  std::memset(p,0xcd,size);
  allocations.push_back(Allocation{p,size,host,true});
  *ptr=p;
  return cudaSuccess;
}
static cudaError_t cudaMalloc(void** ptr,size_t size) { return allocate(ptr,size,false); }
static cudaError_t cudaHostAlloc(void** ptr,size_t size,unsigned flags) {
  require(flags==cudaHostAllocDefault,"unexpected host allocation flags");
  return allocate(ptr,size,true);
}
static cudaError_t cudaFree(void* ptr) {
  Allocation& a=allocation(ptr,false);a.alive=false;std::free(ptr);return cudaSuccess;
}
static cudaError_t cudaFreeHost(void* ptr) {
  Allocation& a=allocation(ptr,true);a.alive=false;std::free(ptr);return cudaSuccess;
}
static cudaError_t cudaMemcpyAsync(void* dst,const void* src,size_t bytes,
                                  int kind,cudaStream_t stream) {
  cudaError_t e=api("copy");
  if (e) return e;
  require(stream && kind==cudaMemcpyDeviceToHost,"wrong copy stream or direction");
  require(bytes==65*sizeof(uint32_t),"readback must contain counter plus 64 hits");
  require(allocation(src,false).size>=bytes,"device read outside allocation");
  require(allocation(dst,true).size==bytes,"host report has wrong capacity");
  ++copyCount;
  stream->work.push_back([=](){std::memcpy(dst,src,bytes);});
  return cudaSuccess;
}
static cudaError_t cudaEventRecord(cudaEvent_t event,cudaStream_t stream) {
  cudaError_t e=api("record");
  if (e) return e;
  require(event && stream,"invalid completion event or stream");
  event->stream=stream;event->fence=stream->work.size();++event->generation;
  ++recordCount;
  return cudaSuccess;
}
static void drain(cudaEvent_t event) {
  require(event && event->stream && event->generation,"drain of unrecorded event");
  MockStream& stream=*event->stream;
  while (stream.executed<event->fence) stream.work[stream.executed++]();
}
static void drainAll(MockStream& stream) {
  while (stream.executed<stream.work.size()) stream.work[stream.executed++]();
}
static void reset() {
  for (const Allocation& a:allocations) require(!a.alive,"allocation leaked");
  allocations.clear();calls.clear();failAt=0;copyCount=recordCount=0;
}
"""

CPP_TEST = r"""
#include "SlotReadback.h"
#include <iostream>
#include <type_traits>
static_assert(!std::is_copy_constructible<qsb::SlotReadback>::value,"copy construction enabled");
static_assert(!std::is_copy_assignable<qsb::SlotReadback>::value,"copy assignment enabled");
static uint32_t hit(unsigned batch,unsigned slot,unsigned index) {
  return ((index&1u)<<30) | ((batch*7919u+slot*104729u+index*17u)&0x3fffffffu);
}
static void produce(qsb::SlotReadback& report,MockStream& stream,
                    unsigned batch,unsigned slot,uint32_t count) {
  uint32_t* counter=report.device_count();
  uint32_t* indices=report.device_indices();
  require(allocation(counter,false).size==1025*sizeof(uint32_t),"device capacity changed");
  // Write all 1024 entries, including the final valid device index. Assign the
  // count last so an overlapping indices pointer cannot disguise corruption.
  stream.work.push_back([=](){
    for (unsigned i=0;i<1024;i++) indices[i]=hit(batch,slot,i);
    *counter=count;
  });
}
static void check(const qsb::SlotReadback& report,unsigned batch,unsigned slot,uint32_t count) {
  require(report.count()==count,"counter mismatch");
  for (unsigned i=0;i<64;i++)
    require(report.indices()[i]==hit(batch,slot,i),"hit prefix mismatch");
  // Same publication cap as the existing slot loop, including count > 64.
  require(std::min(report.count(),uint32_t(64))==std::min(count,uint32_t(64)),
          "publication cap changed");
}
static void schedules() {
  const uint32_t counts[]={0,1,63,64,65,1024};
  {
    qsb::SlotReadback reports[2];MockStream streams[2];MockEvent done[2];
    for (unsigned slot=0;slot<2;slot++) {
      require(reports[slot].init()==cudaSuccess,"init failed");
      const size_t before=calls.size();
      require(reports[slot].init()==cudaErrorInvalidValue,"second init accepted");
      require(calls.size()==before,"second init allocated resources");
    }
    for (unsigned batch=0;batch<18;batch++) {
      for (unsigned slot=0;slot<2;slot++) {
        const uint32_t count=counts[(batch+slot*3)%6];
        produce(reports[slot],streams[slot],batch,slot,count);
        const size_t before=calls.size();
        require(reports[slot].enqueue(&streams[slot],&done[slot])==cudaSuccess,"enqueue failed");
        require(calls.size()==before+2 && calls[before]=="copy" && calls[before+1]=="record",
                "readback must enqueue one copy before its completion record");
        require(done[slot].stream==&streams[slot],"event recorded on wrong slot stream");
        require(done[slot].generation==batch+1,"event generation not renewed");
      }
      const size_t untouched=streams[0].executed;
      drain(&done[1]);check(reports[1],batch,1,counts[(batch+3)%6]);
      require(streams[0].executed==untouched,"drain serialized unrelated slot");
      if (batch) check(reports[0],batch-1,0,counts[(batch-1)%6]);
      drain(&done[0]);check(reports[0],batch,0,counts[batch%6]);
    }
    require(copyCount==36 && recordCount==36,"incorrect readback operation count");
  }
  reset();
  std::cout<<"OK 36 batches, two reused slots, capacities 1024/64, isolated event generations\n";
}
static void errors() {
  for (int failing=1;failing<=2;failing++) {
    reset();
    {
      qsb::SlotReadback report;failAt=failing;
      require(report.init()==injectedError,"allocation error swallowed");
      require(calls.size()==static_cast<size_t>(failing),"init continued after allocation failure");
    }
    reset();
  }
  for (int failing=1;failing<=2;failing++) {
    reset();
    {
      qsb::SlotReadback report;MockStream stream;MockEvent done;
      require(report.init()==cudaSuccess,"error-case init failed");
      produce(report,stream,0,0,64);
      require(report.enqueue(&stream,&done)==cudaSuccess,"initial enqueue failed");
      drain(&done);check(report,0,0,64);
      produce(report,stream,1,0,65);
      const size_t before=calls.size();failAt=static_cast<int>(before)+failing;
      require(report.enqueue(&stream,&done)==injectedError,"enqueue error swallowed");
      require(calls.size()==before+failing,"enqueue continued after failure");
      require(done.generation==1,"failure replaced the old completion generation");
      drain(&done);check(report,0,0,64);
      drainAll(stream); // Complete any accepted copy before destroying buffers.
    }
    reset();
  }
  {
    qsb::SlotReadback report;MockStream stream;MockEvent done;
    require(report.device_count()==nullptr && report.device_indices()==nullptr && report.indices()==nullptr,
            "uninitialized pointer access");
    require(report.enqueue(&stream,&done)==cudaErrorInvalidValue,"uninitialized enqueue accepted");
    require(calls.empty(),"uninitialized enqueue made CUDA calls");
  }
  reset();
  std::cout<<"OK four injected CUDA failures, partial cleanup, stale events, invalid initialization\n";
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"one test mode required");
    if (std::string(argv[1])=="schedules") schedules();
    else if (std::string(argv[1])=="errors") errors();
    else throw std::runtime_error("unknown mode");
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<"\n";return 1; }
}
"""


class SlotReadbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="qsb-slot-readback-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        (cls.root / "cuda_runtime.h").write_text(CUDA_MOCK)
        (cls.root / "test.cc").write_text(CPP_TEST)
        cls.binary = cls.build("actual", HERE)

    @classmethod
    def build(cls, name, header_dir):
        binary = cls.root / name
        subprocess.run(
            [os.environ.get("CXX", "g++"), "-std=c++11", "-O1", "-Wall", "-Wextra",
             "-Werror", "-fsanitize=undefined", "-fno-sanitize-recover=undefined",
             "-I", str(header_dir), "-I", str(cls.root), str(cls.root / "test.cc"),
             "-o", str(binary)], check=True, capture_output=True, text=True,
        )
        return binary

    def run_mode(self, mode):
        result = subprocess.run([str(self.binary), mode], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_prefix_capacity_isolation_and_reuse(self):
        self.assertIn("36 batches", self.run_mode("schedules"))

    def test_errors_and_partial_allocation_cleanup(self):
        self.assertIn("four injected CUDA failures", self.run_mode("errors"))

    def test_overlapping_counter_and_indices_is_detected(self):
        header = (HERE / "SlotReadback.h").read_text()
        original = "return device_ ? device_ + 1 : nullptr;"
        self.assertEqual(header.count(original), 1, "negative control needs updating")
        directory = self.root / "overlap"
        directory.mkdir()
        (directory / "SlotReadback.h").write_text(
            header.replace(original, "return device_ ? device_ + 0 : nullptr;")
        )
        binary = self.build("overlap_test", directory)
        result = subprocess.run([str(binary), "schedules"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, "overlapping indices escaped checker")
        self.assertIn("hit prefix mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
