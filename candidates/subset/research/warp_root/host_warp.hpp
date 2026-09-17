#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
namespace qsb_hm43_host {
// CPU rendezvous model; never a model of CUDA scheduling or memory latency.
struct Warp {
 std::mutex mutex;
 std::condition_variable cv;
 uint64_t generation=0, exchanges=0, ballots=0, joins=0;
 uint32_t arrived=0, observed_mask=0xffffffffu;
 int kind=-1;
 std::array<uint64_t,32> inputs{},outputs{},lane_calls{};
 std::array<int,32> sources{};
 uint64_t collective(int lane,int op,uint64_t value,int source){
  if(lane<0||lane>=32||source<0||source>=32){fprintf(stderr,"COLLECTIVE_BAD_LANE lane=%d source=%d\n",lane,source);std::abort();}
  std::unique_lock<std::mutex> lock(mutex);uint64_t epoch=generation;
  if(arrived==0)kind=op;
  if(kind!=op||(arrived&(1u<<lane))){fprintf(stderr,"COLLECTIVE_SEQUENCE_MISMATCH epoch=%llu lane=%d op=%d expected=%d mask=%08x\n",(unsigned long long)epoch,lane,op,kind,arrived);std::abort();}
  inputs[lane]=value;sources[lane]=source;arrived|=1u<<lane;lane_calls[lane]++;
  if(arrived==0xffffffffu){
   observed_mask&=arrived;
   uint32_t vote=0;
   if(op==1){for(int i=0;i<32;i++)if(inputs[i])vote|=1u<<i;ballots++;}
   if(op==0)exchanges++;
   if(op==2)joins++;
   for(int i=0;i<32;i++)outputs[i]=op==0?inputs[sources[i]]:(op==1?vote:0);
   arrived=0;generation++;cv.notify_all();
  }else if(!cv.wait_for(lock,std::chrono::seconds(10),[&]{return generation!=epoch;})){
   fprintf(stderr,"COLLECTIVE_TIMEOUT epoch=%llu lane=%d kind=%d arrived=%08x (host rendezvous timeout, not GPU evidence)\n",(unsigned long long)epoch,lane,op,arrived);std::abort();
  }
  return outputs[lane];
 }
 uint64_t exchange(int lane,uint64_t value,int source){return collective(lane,0,value,source);}
 uint32_t ballot(int lane,bool value){return (uint32_t)collective(lane,1,value,0);}
 void join(int lane){(void)collective(lane,2,0,0);}
};
inline thread_local Warp *current_warp=nullptr;
inline thread_local int current_lane=-1;
}
inline uint64_t hm43_exchange(uint64_t value,int source){return qsb_hm43_host::current_warp->exchange(qsb_hm43_host::current_lane,value,source);}
inline uint32_t hm43_ballot(bool value){return qsb_hm43_host::current_warp->ballot(qsb_hm43_host::current_lane,value);}
