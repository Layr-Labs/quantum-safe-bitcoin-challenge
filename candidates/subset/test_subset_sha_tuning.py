#!/usr/bin/env python3
"""Check production policy decisions and completed-batch accounting.

All rates in this test are synthetic; none is a GPU measurement.
"""
from pathlib import Path
import subprocess
import tempfile
from test_scalar_producer import function

HERE = Path(__file__).resolve().parent


def main():
    source = r'''
#include "subset_sha_tuning.h"
#include <cassert>
#include <cstdio>
#include <limits>
void scenario(const double *ratios,bool wanted,unsigned invalid=0){
    qsb_subset_sha_trial tune;
    const bool modes[]={false,true,false,true,true,false,true,false,false,true};
    uint64_t searched=0,published=0;
    for(unsigned phase=0;phase<10;phase++){
        assert(!tune.done()&&tune.split()==modes[phase]);
        const unsigned batches=phase<2?2:4;
        uint64_t phase_candidates=0;
        for(unsigned batch=0;batch<batches;batch++){
            // Include a short last batch and zero-hit batches in the model.
            const uint64_t n=batch==3?128*67:134217728;
            searched+=n;phase_candidates+=n;
            // Blocking copy and hit write are complete before policy updates.
            published+=n;
            assert(tune.add_batch(n)==(batch+1==batches));
            assert(tune.split()==modes[phase]);
        }
        assert(searched==published);
        double rate=6e8;
        if(phase>=2&&modes[phase])rate*=ratios[(phase-2)/2];
        double seconds=double(phase_candidates)/rate;
        if(invalid&&phase==4){
            if(invalid==1)seconds=0;
            if(invalid==2)seconds=-1;
            if(invalid==3)seconds=std::numeric_limits<double>::quiet_NaN();
            if(invalid==4)seconds=std::numeric_limits<double>::infinity();
        }
        tune.finish_phase(seconds);
    }
    assert(tune.done()&&tune.split()==wanted);
    assert(!tune.add_batch(1));auto phase=tune.phase;tune.finish_phase(1);
    assert(tune.phase==phase&&tune.split()==wanted);
}
void selection(double paired,double single,unsigned wanted){
    qsb_subset_sha_tuning tune;
    for(unsigned trial=0;trial<qsb_subset_sha_tuning::trials;trial++){
        for(unsigned phase=0;phase<10;phase++){
            unsigned route=qsb_subset_sha_trial::mode(phase)?trial+1:0;
            assert(tune.route()==route);
            unsigned count=phase<2?2:4;
            uint64_t candidates=0;
            for(unsigned batch=0;batch<count;batch++){
                uint64_t n=batch==3?128*67:134217728;
                candidates+=n;
                assert(tune.add_batch(n)==(batch+1==count));
            }
            double gain=route==1?paired:(route==2?single:1.0);
            tune.finish_phase(double(candidates)/(6e8*gain));
        }
    }
    assert(tune.done()&&tune.route()==wanted);
    assert(!tune.add_batch(134217728));tune.finish_phase(1);
    assert(tune.route()==wanted);
}
int main(){
    const double fast[]={1.04,1.03,1.05,1.04};scenario(fast,true);
    const double tie[]={1,1,1,1};scenario(tie,false);
    const double small[]={1.012,1.012,1.012,1.012};scenario(small,false);
    const double slow[]={0.96,0.97,0.96,0.97};scenario(slow,false);
    const double noisy[]={1.06,0.98,1.06,1.06};scenario(noisy,false);
    const double two[]={1.08,1.08,1,1};scenario(two,false);
    const double three[]={1.04,1.04,1.04,1};scenario(three,true);
    for(unsigned invalid=1;invalid<=4;invalid++)scenario(fast,false,invalid);
    for(unsigned stop=0;stop<37;stop++){
        qsb_subset_sha_trial tune;unsigned phases=0;
        for(unsigned batch=0;batch<stop;batch++){
            if(tune.add_batch(134217728)){tune.finish_phase(1);phases++;}
        }
        assert(phases==(stop<4?stop/2:2+(stop-4)/4));
        assert(tune.done()==(stop>=36));
    }
    selection(1.06,1.10,QSB_SUBSET_SHA_SINGLE?2:1);
    selection(1.10,1.06,1);
    selection(1.02,1.08,QSB_SUBSET_SHA_SINGLE?2:0);
    selection(1.08,0.98,1);
    selection(1.02,1.02,0);
    selection(0.96,0.95,0);
    selection(1.06,1.06,1); // Equal passing gains keep the first route.
    const unsigned total=36*qsb_subset_sha_tuning::trials;
    for(unsigned stop=0;stop<=total;stop++){
        qsb_subset_sha_tuning tune;uint64_t searched=0,published=0;
        for(unsigned batch=0;batch<stop;batch++){
            uint64_t n=batch%4==3?128*67:134217728;
            searched+=n;published+=n; // Every completed batch is already drained.
            if(tune.add_batch(n))tune.finish_phase(1);
        }
        assert(searched==published&&tune.done()==(stop==total));
        assert(tune.current==stop/36);
    }
    printf("11 ratio cases; 7 route selections; %u final-batch boundaries; accounting preserved\n",total+1);
}
'''
    with tempfile.TemporaryDirectory(prefix='qsb-subset-tuning-') as tmp:
        cpp, binary = Path(tmp) / 'audit.cpp', Path(tmp) / 'audit'
        cpp.write_text(source)
        for single in (0, 1):
            subprocess.run(['clang++', '-std=c++14', '-O2', '-fsanitize=undefined,bounds',
                            f'-DQSB_SUBSET_SHA_SINGLE={single}',
                            '-I' + str(HERE), str(cpp), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
    tree = (HERE / 'tests/gpu_epochs/tree.cu').read_text()
    loop = tree[tree.index('        qsb_subset_sha_tuning sha_tuning;'):]
    end = loop.index('            struct timespec t_now;')
    loop = loop[:end]
    assert loop.index('kernel_verify_pair_hits<<<') < loop.index('err = cudaMemcpy(zh_host')
    assert loop.index('err = cudaMemcpy(zh_host') < loop.index('write(zh_fd')
    assert loop.index('write(zh_fd') < loop.index('sha_tuning.add_batch')
    assert loop.index('sha_tuning.add_batch') < loop.index('clock_gettime(CLOCK_MONOTONIC,&sha_phase_end)')
    assert loop.index('sha_tuning.finish_phase') < loop.index('sha_route=sha_tuning.route()')
    assert 'Hit tail read failed' in loop
    assert 'if(sha_route==2)' in loop
    assert 'kernel_subset_scalars_single<<<4u*tile.blocks,128,0,scalar_stream>>>(tile_first,d_scalars,tile.epochs)' in loop
    assert 'kernel_subset_scalars<<<tile.blocks,QSB_SE_BLOCK,0,scalar_stream>>>(tile_first,d_scalars,tile.epochs)' in loop
    assert 'if(scalar_error!=cudaSuccess){\n            d_scalars=nullptr;' in tree
    assert 'if(d_scalars && sha_tuning.add_batch' in loop
    def args(mode):
        grid='tile.blocks' if mode=='true' else 'nblk'
        stream=',0,scalar_stream' if mode=='true' else ''
        marker=f'kernel_digest<{mode}><<<{grid}, QSB_SE_BLOCK{stream}>>>('
        start=loop.index(marker)+len(marker)
        return loop[start:loop.index(');',start)]
    tiled=args('true').replace('(int)(tile.blocks*QSB_SE_BLOCK)', 'batch_pos').replace('tile_epochs, tile_first, (int)tile.epochs, d_scalars, tile.first_epoch', 'd_epochs, d_first, epochs_in_batch, d_scalars')
    assert tiled==args('false'), 'unexpected dispatch argument difference'
    assert loop.index('first_block+=tile.blocks')<loop.index('tile_graphs.run')<loop.index('kernel_verify_pair_hits<<<')
    assert '(unsigned)epochs_in_batch==(unsigned)QSB_SE_LAUNCH_BLOCKS*QSB_PAIR_MUL,launch_tiles' in loop
    assert tree.count('(hit_epoch+hit_epoch_offset)*QSB_SE_WINDOWS+hit_lane')==2
    assert tree.count('qsb_pair_front3_z_value(') == 2
    base = '7c3609b87b9d8e094a16be148fe846dfd5ac7807'
    field = subprocess.check_output(['git', 'show', base+':candidates/subset/hit_filter_field_sc.cuh'], cwd=HERE, text=True)
    old_tree = subprocess.check_output(['git', 'show', base+':candidates/subset/tests/gpu_epochs/tree.cu'], cwd=HERE, text=True)
    current_field = (HERE / 'hit_filter_field_sc.cuh').read_text()
    for original, marker in [
        (field, 'template<bool DEFER_Y>\n__device__ __forceinline__ void qsb_filter_point_add('),
        (field, '__device__ void qsb_filter_point_seed('),
        (old_tree, '__device__ __forceinline__ void qsb_filter_last_add('),
        (old_tree, '__device__ void qsb_filter_chain_trial('),
    ]:
        current = current_field if original is field else tree
        assert function(current, marker) == function(original, marker), marker
    print('Source checks: identical dispatch arguments, completed verification/output before timing, allocation fallback')
    print('Runtime fused and produced routes use the four exact promoted arithmetic function bodies')


if __name__ == '__main__':
    main()
