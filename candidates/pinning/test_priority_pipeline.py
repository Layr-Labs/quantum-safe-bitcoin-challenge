#!/usr/bin/env python3
"""CPU dependency tests for the actual CompletionLane header.

The CUDA mock records stream order, event generations and host waits as a DAG.
It does not simulate GPU arithmetic, scheduling priority or GPU performance.
The batch driver models the production launch/drain protocol; the header itself
is included unchanged. Temporary mutated headers prove missing waits are caught.
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
#include <stdexcept>
#include <string>
#include <vector>
typedef int cudaError_t;
const int cudaSuccess=0, cudaErrorInvalidValue=1, injectedError=73;
const unsigned cudaStreamNonBlocking=1, cudaEventDisableTiming=2;
struct MockStream { int last, priority; bool alive; };
struct MockEvent { int last; bool alive; };
typedef MockStream* cudaStream_t;
typedef MockEvent* cudaEvent_t;
struct Node { std::string label; std::vector<int> deps; };
static std::vector<Node> graph;
static std::vector<cudaStream_t> streams;
static std::vector<cudaEvent_t> events;
static int hostFence=-1, calls=0, failAt=0, priorityQueries=0, waits=0;
static int greatestPriority=-3;
static std::vector<std::string> apiCalls;
static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
static cudaError_t api(const char* name) {
  apiCalls.push_back(name);
  return ++calls==failAt ? injectedError : cudaSuccess;
}
static int node(cudaStream_t stream, const std::string& label, int extra=-1) {
  Node n; n.label=label;
  if (stream) {
    require(stream->alive,"operation on destroyed stream");
    if (stream->last>=0) n.deps.push_back(stream->last);
  }
  if (hostFence>=0) n.deps.push_back(hostFence);
  if (extra>=0) n.deps.push_back(extra);
  graph.push_back(n);
  const int index=static_cast<int>(graph.size())-1;
  if (stream) stream->last=index;
  return index;
}
static bool precedes(int before, int after) {
  if (before<0 || after<0) return false;
  std::vector<int> todo(1,after);
  std::vector<bool> seen(graph.size(),false);
  while (!todo.empty()) {
    const int at=todo.back(); todo.pop_back();
    if (at==before) return true;
    if (seen[at]) continue;
    seen[at]=true;
    for (int dependency:graph[at].deps) todo.push_back(dependency);
  }
  return false;
}
static void ordered(int before,int after,const char* message) {
  require(precedes(before,after),message);
}
static cudaError_t cudaDeviceGetStreamPriorityRange(int* least,int* greatest) {
  cudaError_t e=api("priority-range");
  if (!e) { ++priorityQueries; *least=0; *greatest=greatestPriority; }
  return e;
}
static cudaError_t cudaStreamCreateWithPriority(cudaStream_t* output,unsigned flags,int priority) {
  require(flags==cudaStreamNonBlocking,"auxiliary stream must be nonblocking");
  cudaError_t e=api("create-stream");
  if (!e) { *output=new MockStream{-1,priority,true}; streams.push_back(*output); }
  return e;
}
static cudaError_t cudaEventCreateWithFlags(cudaEvent_t* output,unsigned flags) {
  require(flags==cudaEventDisableTiming,"dependency events must disable timing");
  cudaError_t e=api("create-event");
  if (!e) { *output=new MockEvent{-1,true}; events.push_back(*output); }
  return e;
}
static cudaError_t cudaEventRecord(cudaEvent_t event,cudaStream_t stream) {
  cudaError_t e=api("record");
  if (!e) {
    require(event && event->alive,"record requires live event");
    event->last=node(stream,"event-record");
  }
  return e;
}
static cudaError_t cudaStreamWaitEvent(cudaStream_t stream,cudaEvent_t event,unsigned flags) {
  require(flags==0,"unexpected wait flags");
  cudaError_t e=api("wait");
  if (!e) {
    require(event && event->alive && event->last>=0,"wait must follow event record");
    ++waits;
    // Capture this generation, never a mutable pointer to a future recording.
    node(stream,"event-wait",event->last);
  }
  return e;
}
static cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  cudaError_t e=api("synchronize");
  if (!e) {
    require(event && event->alive && event->last>=0,"synchronize unrecorded event");
    hostFence=node(nullptr,"host-drain",event->last);
  }
  return e;
}
static cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  require(stream && stream->alive,"duplicate stream destruction");
  stream->alive=false;
  return cudaSuccess;
}
static cudaError_t cudaEventDestroy(cudaEvent_t event) {
  require(event && event->alive,"duplicate event destruction");
  event->alive=false;
  return cudaSuccess;
}
static void checkNoResources() {
  for (cudaStream_t s:streams) require(!s->alive,"leaked stream");
  for (cudaEvent_t e:events) require(!e->alive,"leaked event");
}
static void resetMock() {
  for (cudaStream_t s:streams) delete s;
  for (cudaEvent_t e:events) delete e;
  streams.clear(); events.clear(); graph.clear(); apiCalls.clear();
  hostFence=-1; calls=failAt=priorityQueries=waits=0; greatestPriority=-3;
}
"""

CPP_TEST = r"""
#include "PriorityPipeline.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

static cudaStream_t prepareStream() {
  cudaStream_t s=nullptr;
  require(cudaStreamCreateWithPriority(&s,cudaStreamNonBlocking,0)==cudaSuccess,"prepare create");
  return s;
}
struct Batch {
  int sequence,slot;
  uint64_t offset,count;
  int upload,clear,prepare,rootPrepare,invert,rootFinish,finish,copyCount,copyHits,done;
};
static void schedule(int mode,int slotCount,uint64_t range,uint64_t width,bool offload,bool noPriority) {
  require(range && width && slotCount>0,"invalid test geometry");
  if (noPriority) greatestPriority=0;
  std::vector<cudaStream_t> prepare(slotCount);
  std::vector<cudaEvent_t> done(slotCount);
  std::vector<bool> busy(slotCount,false);
  std::vector<int> occupant(slotCount,-1);
  std::vector<Batch> batches;
  uint64_t searched=0,drained=0,batchNo=0;
  int lastReplacement=-1;
  {
    std::unique_ptr<qsb::CompletionLane[]> lane(new qsb::CompletionLane[slotCount]);
    for (int s=0;s<slotCount;++s) {
      prepare[s]=prepareStream();
      require(lane[s].init(prepare[s],mode)==cudaSuccess,"lane init");
      require(cudaEventCreateWithFlags(&done[s],cudaEventDisableTiming)==cudaSuccess,"done create");
    }
    auto drain=[&](int s) {
      if (!busy[s]) return;
      require(cudaEventSynchronize(done[s])==cudaSuccess,"drain failed");
      const Batch& b=batches[occupant[s]];
      ordered(b.copyHits,hostFence,"host reads hits before copy completion");
      ordered(b.finish,hostFence,"host drains unfinished batch");
      drained+=b.count; busy[s]=false;
    };
    for (int sequence=0;sequence<3;++sequence) {
      int firstPrepare=-1;
      uint64_t nextExpected=0;
      for (uint64_t offset=0;offset<range;offset+=width) {
        const int s=static_cast<int>(batchNo++%slotCount);
        const int prior=occupant[s];
        drain(s);
        Batch b={}; b.sequence=sequence;b.slot=s;b.offset=offset;b.count=std::min(width,range-offset);
        require(b.offset==nextExpected,"candidate gap or overlap");
        nextExpected+=b.count;
        cudaStream_t st=prepare[s];
        b.upload=node(st,"midstate-upload");
        if (lastReplacement>=0)
          ordered(lastReplacement,b.upload,"new sequence starts before constants replacement");
        b.clear=node(st,"hit-counter-clear");
        b.prepare=node(st,"prepare");
        if (prior>=0) ordered(batches[prior].copyHits,b.upload,"slot reused before old hits copied");
        if (!sequence && batches.empty()) firstPrepare=b.prepare;
        if (!sequence && batches.size()==1 && slotCount>1)
          require(!precedes(firstPrepare,b.prepare) && !precedes(b.prepare,firstPrepare),"independent slots serialized");
        int lastProducer=b.prepare;
        if (offload) lastProducer=node(st,"leaf-prepare");
        require(lane[s].begin_roots(st)==cudaSuccess,"begin roots");
        if (mode==0) require(st==prepare[s],"baseline switched stream");
        else {
          require(st!=prepare[s],"split mode did not switch stream");
          require(st->priority==(mode==3 ? 0 : greatestPriority),"wrong root priority");
        }
        b.rootPrepare=node(st,"root-prepare");
        b.invert=node(st,"invert");
        b.rootFinish=node(st,"root-finish");
        ordered(lastProducer,b.rootPrepare,"roots can run before prepare producer");
        require(lane[s].end_roots(st)==cudaSuccess,"end roots");
        require((st==prepare[s])==(mode==0 || mode==1),"wrong finish stream");
        int firstConsumer=-1;
        if (offload) firstConsumer=node(st,"leaf-finish");
        b.finish=node(st,"finish");
        if (firstConsumer<0) firstConsumer=b.finish;
        ordered(b.rootFinish,firstConsumer,"finish can run before root producer");
        st=lane[s].completion_stream();
        b.copyCount=node(st,"copy-count");
        b.copyHits=node(st,"copy-hits");
        require(cudaEventRecord(done[s],st)==cudaSuccess,"record done");
        b.done=done[s]->last;
        const int chain[]={b.upload,b.clear,b.prepare,b.rootPrepare,b.invert,b.rootFinish,b.finish,b.copyCount,b.copyHits,b.done};
        for (unsigned i=1;i<sizeof(chain)/sizeof(chain[0]);++i)
          ordered(chain[i-1],chain[i],"batch data dependency missing");
        occupant[s]=static_cast<int>(batches.size()); batches.push_back(b);
        busy[s]=true;searched+=b.count;
      }
      require(nextExpected==range,"tail count changed");
      for (int s=0;s<slotCount;++s) drain(s);
      require(drained==searched,"sequence rollover before all hits drained");
      lastReplacement=node(nullptr,"sequence-constants-replacement");
      hostFence=lastReplacement; // Synchronous per-sequence constant upload.
    }
    require(searched==range*3 && drained==searched,"candidate total mismatch");
    const int expectedWaits=mode==0 ? 0 : static_cast<int>(batches.size())*(mode==1 ? 2 : 1);
    require(waits==expectedWaits,"wrong number of cross-stream dependencies");
    for (int s=0;s<slotCount;++s) {
      cudaEventDestroy(done[s]);cudaStreamDestroy(prepare[s]);
    }
  }
  checkNoResources();
  std::cout<<"OK batches="<<batches.size()<<" candidates="<<searched<<" waits="<<waits<<"\n";
}

static void errors() {
  for (int mode=0;mode<4;++mode) {
    const int initCalls=mode==0 ? 0 : mode==1 ? 4 : 3;
    for (int failing=1;failing<=initCalls;++failing) {
      resetMock();cudaStream_t p=prepareStream();
      {
        qsb::CompletionLane lane;
        failAt=calls+failing;
        require(lane.init(p,mode)==injectedError,"init swallowed CUDA error");
        require(calls==failAt,"init continued after CUDA failure");
      }
      cudaStreamDestroy(p);checkNoResources();
    }
    for (int phase=0;phase<2;++phase) {
      const int phaseCalls=(phase==0 ? mode!=0 : mode==1) ? 2 : 0;
      for (int failing=1;failing<=phaseCalls;++failing) {
        resetMock();cudaStream_t p=prepareStream();
        {
          qsb::CompletionLane lane;require(lane.init(p,mode)==cudaSuccess,"init error test");
          cudaStream_t st=p;node(st,"prepare");
          if (phase==1) { require(lane.begin_roots(st)==cudaSuccess,"begin before error");node(st,"roots"); }
          cudaStream_t before=st;
          failAt=calls+failing;
          cudaError_t result=phase==0 ? lane.begin_roots(st) : lane.end_roots(st);
          require(result==injectedError,"handoff swallowed CUDA error");
          require(st==before,"handoff exposed new stream after failure");
          require(calls==failAt,"handoff continued after failure");
        }
        cudaStreamDestroy(p);checkNoResources();
      }
    }
  }
  for (int bad:std::vector<int>{-1,4,100}) {
    resetMock();cudaStream_t p=prepareStream();
    {
      qsb::CompletionLane lane;int before=calls;
      require(lane.init(p,bad)==cudaErrorInvalidValue,"invalid mode accepted");
      require(calls==before,"invalid mode made CUDA calls");
    }
    cudaStreamDestroy(p);checkNoResources();
  }
  resetMock();cudaStream_t p=prepareStream();
  {
    qsb::CompletionLane lane;int before=calls;
    require(lane.init(p,0)==cudaSuccess,"baseline init");cudaStream_t st=p;
    require(lane.begin_roots(st)==cudaSuccess && lane.end_roots(st)==cudaSuccess,"baseline handoff");
    require(calls==before && st==p && lane.completion_stream()==p,"baseline adds CUDA work");
  }
  cudaStreamDestroy(p);checkNoResources();
  std::cout<<"OK error propagation and cleanup\n";
}
int main(int argc,char** argv) {
  try {
    if (argc==2 && std::string(argv[1])=="errors") errors();
    else {
      require(argc==7,"schedule arguments");
      schedule(std::atoi(argv[1]),std::atoi(argv[2]),std::strtoull(argv[3],nullptr,10),
               std::strtoull(argv[4],nullptr,10),std::atoi(argv[5])!=0,std::atoi(argv[6])!=0);
    }
    resetMock();return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<"\n";return 1; }
}
"""


class PriorityPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="qsb-priority-dag-")
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

    @staticmethod
    def run_schedule(binary, mode, slots, count, batch, offload=0, no_priority=0):
        return subprocess.run(
            [str(binary), str(mode), str(slots), str(count), str(batch),
             str(offload), str(no_priority)], capture_output=True, text=True,
        )

    def test_dependency_paths_slots_reuse_partial_batches_and_rollover(self):
        for mode in range(4):
            for slots in (1, 2, 4):
                for count in (1, 8, 23):
                    for offload in (0, 1):
                        with self.subTest(mode=mode, slots=slots, count=count, offload=offload):
                            result = self.run_schedule(self.binary, mode, slots, count, 8, offload)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertIn(f"candidates={count * 3}", result.stdout)

    def test_real_ranked_range_and_batch_tail(self):
        for mode in range(4):
            with self.subTest(mode=mode):
                result = self.run_schedule(self.binary, mode, 2, 1_244_600_000, 8_388_608)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("batches=447 candidates=3733800000", result.stdout)

    def test_devices_without_priority_differentiation_remain_correct(self):
        for mode in (1, 2, 3):
            with self.subTest(mode=mode):
                result = self.run_schedule(self.binary, mode, 2, 23, 8, no_priority=1)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_errors_invalid_modes_and_partial_initialization_cleanup(self):
        result = subprocess.run([str(self.binary), "errors"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_cross_stream_waits_are_detected(self):
        header = (HERE / "PriorityPipeline.h").read_text()
        cases = (
            ("prepare_wait", "e = cudaStreamWaitEvent(auxiliary_, prepared_, 0)", 2,
             "roots can run before prepare producer"),
            ("return_wait", "e = cudaStreamWaitEvent(prepare_, roots_ready_, 0)", 1,
             "finish can run before root producer"),
        )
        for name, original, mode, diagnostic in cases:
            with self.subTest(mutation=name):
                self.assertEqual(header.count(original), 1, "negative control needs updating")
                directory = self.root / name
                directory.mkdir()
                (directory / "PriorityPipeline.h").write_text(header.replace(original, "e = cudaSuccess"))
                binary = self.build(name + "_test", directory)
                result = self.run_schedule(binary, mode, 2, 23, 8)
                self.assertNotEqual(result.returncode, 0, "missing wait escaped dependency checker")
                self.assertIn(diagnostic, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
