// Host concurrency oracle: exercise the actual qhp alias-reuse waiter.
#define QSB_HP_CHECK_ALIAS 1
#define main qsb_candidate_main
#include "tree.cu"
#undef main
#include <atomic>
int main(){
 qhp::Hp h;qhp::g_hp=&h;h.chk_enqueued=true;h.d_scr_ep=(uint8_t*)0x1234;
 std::atomic<int>done{0};
 auto run=[&]{done=0;std::thread t([&]{qhp::wait_check_reuse(h.d_scr_ep);done=1;});
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  if(done){t.join();return false;}
  {std::lock_guard<std::mutex>g(h.m);h.chk_running=false;h.cv_ready.notify_all();}
  t.join();return done.load()==1;};
 h.dead=true;h.stop=true;h.check=0;h.chk_running=true;
 if(!run())return 2;
 h.dead=false;h.stop=false;h.check=1;h.chk_running=true;
 if(!run())return 3;
 h.check=-1;h.chk_running=false;qhp::wait_check_reuse(h.d_scr_ep);
 h.check=0;h.chk_running=true;qhp::wait_check_reuse((void*)0x4321);
 qhp::g_hp=nullptr;printf("ALIAS_REUSE_CONCURRENT_FAILURE_AUDIT: PASS cases=4\n");return 0;
}
