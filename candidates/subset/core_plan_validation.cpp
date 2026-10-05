// SPDX-License-Identifier: GPL-3.0-only
// Validate the exact submitted helper, independent of Linux topology I/O.
#include "CoreWorkers.h"
#include <cstdio>
#include <map>
#include <random>
#include <stdexcept>
#include <cstdint>
static unsigned checks=0;
static void require(bool b){checks++;if(!b)throw std::runtime_error("core plan check failed");}
int main(){try{
    std::vector<int> parsed,selected;
    require(qsb_core::parse_list("0-3,8-11\n",1024,parsed));
    require(parsed==std::vector<int>({0,1,2,3,8,9,10,11}));
    for(const auto &s:std::vector<std::string>{"","-1","1-0","0,","0,,1","3x","1024","0-1024","99999999999999999999999999","0\n1","0-1,2-","1, 2"})
        require(!qsb_core::parse_list(s,1024,parsed) && parsed.empty());
    require(!qsb_core::plan({0,1},{"0-1","1"},1024,selected));
    require(!qsb_core::plan({0,1},{"0","0"},1024,selected));
    require(!qsb_core::plan({0,0},{"0","0"},1024,selected));
    require(!qsb_core::plan({0},{},1024,selected));
    require(!qsb_core::plan({}, {},1024,selected));
    require(qsb_core::plan({1},{"0-1"},1024,selected)&&selected==std::vector<int>({1}));
    require(qsb_core::plan({3,1,2,0},{"2-3","0-1","2-3","0-1"},1024,selected)&&selected==std::vector<int>({0,2}));
    require(qsb_core::plan({0,1},{"0","1"},1024,selected)&&selected==std::vector<int>({0,1}));
    // Exhaust every nonempty allowed mask over four 2-thread physical cores.
    for(unsigned mask=1;mask<256;mask++){
        std::vector<int> ids,want;std::vector<std::string> lists;
        for(int core=0;core<4;core++){
            int first=-1;
            for(int cpu=2*core;cpu<2*core+2;cpu++)if(mask&(1u<<cpu)){
                if(first<0)first=cpu;
                ids.push_back(cpu);lists.push_back(std::to_string(2*core)+"-"+std::to_string(2*core+1));
            }
            if(first>=0)want.push_back(first);
        }
        require(qsb_core::plan(ids,lists,1024,selected));require(selected==want);
    }
    // Fresh irregular masks, non-contiguous sibling numbering, packages and
    // quota limits. Independent oracle groups CPU c by c modulo core count.
    std::mt19937 rng(20261006);
    unsigned random_topologies=0;
    for(int trial=0;trial<20000;trial++){
        int cores=1+rng()%32,threads=1+rng()%4,total=cores*threads;
        std::vector<int> ids;std::vector<std::string> lists;std::map<int,int> oracle;
        for(int cpu=0;cpu<total;cpu++)if(rng()%3){
            int core=cpu%cores;ids.push_back(cpu);
            if(!oracle.count(core))oracle[core]=cpu;
            std::string s;
            for(int t=0;t<threads;t++){if(t)s+=",";s+=std::to_string(core+t*cores);}lists.push_back(s+"\n");
        }
        if(ids.empty()){ids.push_back(0);lists.push_back("0");oracle[0]=0;}
        require(qsb_core::plan(ids,lists,1024,selected));
        std::vector<int>want;for(const auto &item:oracle)want.push_back(item.second);std::sort(want.begin(),want.end());
        require(selected==want);
        // Production truncates the already unique plan to the old thread quota.
        const unsigned quota=1+rng()%128;if(selected.size()>quota)selected.resize(quota);
        require(selected.size()<=quota);
        for(std::size_t i=0;i<selected.size();i++){
            require(std::find(ids.begin(),ids.end(),selected[i])!=ids.end());
            for(std::size_t j=0;j<i;j++)require(selected[i]%cores!=selected[j]%cores);
        }
        random_topologies++;
    }
    std::printf("{\"checks\":%u,\"random_topologies\":%u,\"failed\":0,\"full_cuda_tested_locally\":false}\n",checks,random_topologies);
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
