#include "../../CpuWorkerPlacement.h"
#include <assert.h>
#include <initializer_list>
#include <stdio.h>
static int checks;
static void check(bool v) { ++checks; assert(v); }
static cpu_set_t mask(std::initializer_list<int> xs) { cpu_set_t s; CPU_ZERO(&s); for(int x:xs) CPU_SET(x,&s); return s; }
static cpu_set_t cores[CPU_SETSIZE];
static int missing=-1;
static bool fixture(int cpu,cpu_set_t *out) { if(cpu==missing)return false; *out=cores[cpu];return true; }
static cpu_set_t plan(cpu_set_t allowed,cpu_set_t main) { cpu_set_t work;check(qcpu_place::conservative_mask(allowed,main,&work,fixture));return work; }
int main() {
 cpu_set_t parsed;
 for(const char *s:{"2,18\n","2,18","2,18 \n"}) {check(qcpu_place::parse_core(s,2,&parsed)); auto expected=mask({2,18});check(CPU_EQUAL(&parsed,&expected));}
 check(qcpu_place::parse_core("2-3\n",2,&parsed)); check(CPU_COUNT(&parsed)==2);
 check(qcpu_place::parse_core("2\n",2,&parsed)); check(CPU_COUNT(&parsed)==1);
 for(const char *s:{"","2,","2,,18","2,-18","2--18","18-2","2,2","1024","-1","2x","2 18","3","2-1024","2,18junk","999999999999999999999"}) check(!qcpu_place::parse_core(s,2,&parsed));
 for(int i=0;i<CPU_SETSIZE;++i) cores[i]=mask({i});
 cpu_set_t allowed;CPU_ZERO(&allowed);for(int i=0;i<30;++i)CPU_SET(i,&allowed);
 auto work=plan(allowed,mask({7}));check(CPU_COUNT(&work)==29);check(!CPU_ISSET(7,&work)); // W1: default count stays 30-2=28.
 for(int i=0;i<16;++i)cores[i]=cores[i+16]=mask({i,i+16});
 CPU_ZERO(&allowed);for(int i=0;i<32;++i)CPU_SET(i,&allowed);
 work=plan(allowed,mask({7}));check(CPU_COUNT(&work)==30);check(!CPU_ISSET(7,&work));check(!CPU_ISSET(23,&work));
 work=plan(allowed,mask({7,23}));check(CPU_COUNT(&work)==30);check(!CPU_ISSET(23,&work));
 // A partial cpuset can still safely exclude any allowed proven sibling.
 auto partial=mask({0,1,7,16,17});work=plan(partial,mask({7}));check(CPU_COUNT(&work)==4);
 missing=23;work=plan(allowed,mask({7}));check(CPU_COUNT(&work)==31);check(!CPU_ISSET(7,&work));check(CPU_ISSET(23,&work));missing=-1;
 cores[23]=mask({23});work=plan(allowed,mask({7}));check(CPU_COUNT(&work)==31);check(CPU_ISSET(23,&work));
 // Failure on a later main CPU rolls back all earlier sibling exclusions.
 work=plan(allowed,mask({0,7}));check(CPU_COUNT(&work)==30);check(CPU_ISSET(16,&work));check(CPU_ISSET(23,&work));
 check(!qcpu_place::conservative_mask(allowed,allowed,&work,fixture));
 auto empty=mask({});check(!qcpu_place::conservative_mask(allowed,empty,&work,fixture));
 auto outside=mask({33});check(!qcpu_place::conservative_mask(allowed,outside,&work,fixture));
 // No phantom empty CPU mask when one physical core is the entire cpuset.
 cores[0]=cores[16]=mask({0,16});work=plan(mask({0,16}),mask({0}));check(CPU_COUNT(&work)==1);check(CPU_ISSET(16,&work));
 printf("%d conservative placement assertions passed\n",checks);
}
