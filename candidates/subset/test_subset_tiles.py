#!/usr/bin/env python3
"""Audit the actual tile planner over full-run geometry and boundary tails."""
from pathlib import Path
import subprocess,tempfile
HERE=Path(__file__).resolve().parent
CODE=r'''
#include "subset_tile_plan.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
int main(){
    unsigned cases=0;
    for(unsigned windows:{128u,256u}){
        const unsigned pair_mul=512/windows;
        std::vector<unsigned> counts={0,1,2,3,4,5,31,32,33,67,2047,2048,2049,4095,4096,4097,1048573,1048576};
        for(unsigned epochs:counts){
            const unsigned blocks=(epochs+pair_mul-1)/pair_mul;
            unsigned covered_epochs=0,covered_blocks=0;
            uint64_t candidates=0;
            for(unsigned first=0;first<blocks;){
                const auto t=qsb_subset_next_tile(first,blocks,epochs,pair_mul);
                assert(t.first_epoch==covered_epochs&&t.first_block==covered_blocks);
                assert(t.blocks>0&&t.blocks<=qsb_subset_tile_capacity(blocks));
                assert(t.epochs>0&&t.epochs<=t.blocks*pair_mul);
                assert(t.first_epoch+t.epochs<=epochs);
                // Both recovery signs share one global epoch/lane identity.
                for(unsigned e:{0u,t.epochs-1})for(unsigned lane:{0u,windows-1})for(unsigned recid:{0u,1u}){
                    const uint32_t id=(t.first_epoch+e)*windows+lane;
                    const uint32_t tag=id|(recid<<30);
                    assert(((tag&0x3fffffffu)/windows)==t.first_epoch+e);
                    assert((tag&(windows-1))==lane&&((tag>>30)&1)==recid);
                }
                candidates+=(uint64_t)t.epochs*windows;
                covered_epochs+=t.epochs;covered_blocks+=t.blocks;first+=t.blocks;
            }
            assert(covered_epochs==epochs&&covered_blocks==blocks);
            assert(candidates==(uint64_t)epochs*windows);++cases;
        }
    }
    printf("%u planner cases; tile cap %u; exact coverage and tagged identities\n",cases,qsb_subset_tile_capacity(262144));
}
'''
def main():
    with tempfile.TemporaryDirectory(prefix='qsb-tile-plan-') as tmp:
        p=Path(tmp);(p/'audit.cpp').write_text(CODE)
        for enabled,cap in ((0,512),(1,1),(1,3),(1,512),(1,2048)):
            subprocess.run(['clang++','-std=c++17','-O2','-fsanitize=undefined,bounds',
                f'-DQSB_SUBSET_TILED={enabled}',f'-DQSB_SUBSET_TILE_BLOCKS={cap}',
                '-I'+str(HERE),str(p/'audit.cpp'),'-o',str(p/'audit')],check=True)
            subprocess.run([str(p/'audit')],check=True)
if __name__=='__main__':main()
