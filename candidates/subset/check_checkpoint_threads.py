#!/usr/bin/env python3
"""ThreadSanitizer on extracted checkpoint code with CPU warp/block barriers.

This checks C++ happens-before edges, not GPU execution or CUDA race freedom.
Two deliberately missing cross-warp barriers must be rejected by the sanitizer.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from check_candidate import BACKEND
from preflight import source_identity

HERE = Path(__file__).resolve().parent
DRIVER = r'''
int main(){
    uint64_t roots[4]={},checkpoint[4*QSB_CHECKPOINT_STRIDE]={};
    uint64_t inputs[256][5]={},outputs[256][5]={};
    for(int round=0;round<3;round++){
        for(int i=0;i<256;i++)inputs[i][0]=1+i+round*256;
        launch(256,[&](int i){
            if((i+round)%7==0)for(int j=0;j<50;j++)std::this_thread::yield();
            qsb_block_product_checkpoint(inputs[i],roots,checkpoint);
        });
        uint64_t inv[5]={};memcpy(inv,roots,32);_ModInv(inv);memcpy(roots,inv,32);
        launch(256,[&](int i){
            if((i+round)%5==0)for(int j=0;j<50;j++)std::this_thread::yield();
            memcpy(outputs[i],inputs[i],40);
            qsb_block_inverse_checkpoint(outputs[i],roots,checkpoint);
        });
        for(int i=0;i<256;i++){
            uint64_t want[5];memcpy(want,inputs[i],40);_ModInv(want);
            require(memcmp(want,outputs[i],40)==0);
        }
    }
    puts("PASS: 768 checkpoint inverses under CPU ThreadSanitizer");
}
'''


def main():
    identity = source_identity()
    header = (HERE/'tests/gpu_epochs/ranked_pipeline.cuh').read_text()
    header = header[:header.index('/* Batch the per-search-CTA roots')]
    variants = {
        'control': header,
        'up_missing_cross_warp': header.replace('if(count>64)__syncthreads();',
                                                'if(count>128)__syncthreads();'),
        'down_missing_cross_warp': header.replace('if(count>=32)__syncthreads();',
                                                  'if(count>=64)__syncthreads();'),
    }
    assert all(variants[k] != header for k in variants if k != 'control')
    reports = {}
    with tempfile.TemporaryDirectory(prefix='qsb-checkpoint-tsan-') as td:
        for name, body in variants.items():
            cpp, binary = Path(td)/f'{name}.cpp', Path(td)/name
            cpp.write_text(BACKEND + body + DRIVER)
            subprocess.run(['c++', '-std=c++17', '-O1', '-g', '-pthread',
                            '-fsanitize=thread', '-fno-omit-frame-pointer',
                            str(cpp), '-lcrypto', '-o', str(binary)], check=True,
                           stdout=sys.stderr, stderr=sys.stderr)
            env = dict(os.environ, TSAN_OPTIONS='halt_on_error=1:exitcode=66')
            result = subprocess.run([str(binary)], capture_output=True,
                                    text=True, env=env, timeout=90)
            diagnostic = result.stdout + result.stderr
            if name == 'control':
                assert result.returncode == 0, diagnostic
                reports[name] = {'status': 'PASS', 'inverses': 768}
            else:
                assert result.returncode == 66 and 'ThreadSanitizer: data race' in diagnostic, diagnostic
                reports[name] = {'status': 'rejected_data_race',
                                 'diagnostic_sha256': hashlib.sha256(diagnostic.encode()).hexdigest()}
    assert source_identity() == identity
    print(json.dumps({'status': 'PASS', 'cpu_thread_sanitizer': reports,
                      'gpu_executed': False, **identity,
                      'audit_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'limits': 'Extracted C++ code with OpenSSL field operations and CPU barriers; no CUDA sanitizer or GPU execution.'}, indent=2))


if __name__ == '__main__':
    main()
