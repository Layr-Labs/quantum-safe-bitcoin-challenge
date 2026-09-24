#!/usr/bin/env python3
"""Compile isolated cache-policy candidates; retain compact reports and patches."""
import argparse
import collections
import concurrent.futures
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

HELPER = '''__device__ __forceinline__ ulonglong2 qsb_cache_review_cs(const ulonglong2 *ptr) {
    ulonglong2 v;
    asm("{ .reg .u64 g; cvta.to.global.u64 g, %2; ld.global.cs.v2.u64 {%0,%1}, [g]; }"
        : "=l"(v.x), "=l"(v.y) : "l"(ptr));
    return v;
}
'''
LOADER = '__device__ __forceinline__ void gt_load_signed_flat_m('
OLD = 'ulonglong2 x0=__ldg(tx),x1=__ldg(tx+1),y0=__ldg(ty),y1=__ldg(ty+1);'
HOT = 'x0=__ldg(tx);x1=__ldg(tx+1);y0=__ldg(ty);y1=__ldg(ty+1);'
COLD = 'x0=qsb_cache_review_cs(tx);x1=qsb_cache_review_cs(tx+1);y0=qsb_cache_review_cs(ty);y1=qsb_cache_review_cs(ty+1);'


def variants(original):
    clamp = 'size_t want = gt_sz < (size_t)max_persist ? gt_sz : (size_t)max_persist;'
    assert original.count(clamp) == 2
    source = original.replace(clamp, clamp + '\n#if QSB_BIGTBL && QSB_FOUR_HOT\n        if(want > 48u*1024u*1024u) want=48u*1024u*1024u;\n#endif')
    source = source.replace(LOADER, HELPER + LOADER, 1)
    record = source.replace(OLD, 'ulonglong2 x0,x1,y0,y1;\n#if QSB_BIGTBL && QSB_FOUR_HOT\n    if(base+idx >= 786432u) { ' + COLD + ' }\n    else\n#endif\n    { ' + HOT + ' }', 1)
    template = source.replace(LOADER, 'template<bool COLD=false>\n'+LOADER, 1)
    template = template.replace(OLD, 'ulonglong2 x0,x1,y0,y1;\n    if(COLD) { '+COLD+' } else { '+HOT+' }', 1)
    start = template.index('__device__ __forceinline__ void qsb_load_glv(')
    end = template.index('__device__ void _FixedBaseSignedXYZZScalar', start)
    old = template[start:end]
    template = template[:start] + 'template<bool COLD=false>\n' + old.replace('gt_load_signed_flat_m(', 'gt_load_signed_flat_m<COLD>(') + template[end:]
    selector = '''#if QSB_BIGTBL && QSB_FOUR_HOT
        if((term>=4 && term<6) || term>=10) qsb_load_glv<true>(table,term,x1,y1);
        else qsb_load_glv<false>(table,term,x1,y1);
#else
        qsb_load_glv<false>(table,term,x1,y1);
#endif'''
    selected = template.replace('        qsb_load_glv(table,term,x1,y1);', selector, 1)
    start = template.index('    #pragma unroll 1\n    for(int term=first+2;term<last;term++) {')
    end = template.index('\n#if QSB_YOFF\n    qsb_yoff_to_y(y0);', start)
    old_loop = template[start:end]
    phi_start = old_loop.index('            /* phi(')
    phi_end = old_loop.index('\n        }', phi_start)
    phi = old_loop[phi_start:phi_end]

    def loop(limit, cold):
        return '\n    #pragma unroll 1\n    for(;term<'+str(limit)+';term++) {\n        qsb_load_glv<'+str(cold).lower()+'>(table,term,x1,y1);\n        _PointAddXYZZT<true>(X,Y,U,V,x1,y1,y0);\n        Load256(y0,y1);\n    }'

    split_loop = '\n#if !(QSB_BIGTBL && QSB_FOUR_HOT)\n#error cache review split-loop requires four-hot GLV12\n#endif\n    int term=first+2;'+loop(4,False)+loop(6,True)+'\n    if(first==0) {\n'+phi+'\n    }'+loop(10,False)+loop(12,True)
    split = template[:start]+split_loop+template[end:]
    return {'baseline':original, 'record_branch':record, 'term_select':selected, 'split_loops':split}


def parse_sass(text):
    data = {}
    for block in text.split('Function : ')[1:]:
        name, body = block.split('\n', 1)
        if 'kernel_pinning_pipeline' not in name:
            continue
        stage = 'prepare' if 'ILb1ELi0' in name else 'finish'
        data[stage] = (name, re.findall(r'/\*[0-9a-f]+\*/\s+([^;]+);', body))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--cuobjdump', default='cuobjdump')
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    root = out.parents[2]
    original = (root/'pinning.cu').read_text()
    sources = variants(original)
    report = {'scope':'Static compiler screen; no GPU timing', 'source_sha256':hashlib.sha256(original.encode()).hexdigest(), 'flags':['-O3','-DQSB_ZEROS_N=24','-DQSB_GLV_COLD_SEED=0','--cubin','--ptxas-options=-v'], 'variants':{}}
    with tempfile.TemporaryDirectory(prefix='qsb-retry-cache-') as directory:
        temp = Path(directory)
        for name, source in sources.items():
            (temp/(name+'.cu')).write_text(source)

        def build(spec):
            name, arch = spec
            key = name+'_'+arch
            binary = temp/(key+'.cubin')
            flags = ['-arch=sm_89'] if arch=='sm89' else []
            cmd = [args.nvcc]+report['flags']+flags+['-I',str(root),str(temp/(name+'.cu')),'-o',str(binary)]
            proc = subprocess.run(cmd,text=True,capture_output=True)
            if proc.returncode:
                return key, {'returncode':proc.returncode,'error':proc.stderr[-1500:].replace(str(temp),'TEMP').replace(str(root),'candidates/pinning')}
            sass = subprocess.check_output([args.cuobjdump,'--dump-sass',str(binary)],text=True)
            item = {'returncode':0,'kernels':{}}
            for stage, (symbol, instructions) in parse_sass(sass).items():
                log = proc.stderr.split('Function properties for '+symbol,1)[1].split('Compile time',1)[0]
                def count(pattern):
                    match = re.search(pattern,log)
                    return int(match[1]) if match else 0
                counts = collections.Counter(re.sub(r'^@\S+\s+', '', i.strip()).split()[0] for i in instructions)
                value = {'instructions':len(instructions),'registers':count(r'Used (\d+) registers'),'stack':count(r'(\d+) bytes stack frame'),'spill_stores':count(r'(\d+) bytes spill stores'),'spill_loads':count(r'(\d+) bytes spill loads'),'smem':count(r'(\d+) bytes smem')}
                value['sass_sha256'] = hashlib.sha256('\n'.join(instructions).encode()).hexdigest()
                value['loads'] = {k:n for k,n in counts.items() if 'LDG' in k}
                value['control'] = {k:n for k,n in counts.items() if any(s in k for s in ['BRA','BSSY','BSYNC','SSY','SYNC'])}
                value['counts'] = dict(counts)
                if stage=='prepare':
                    value['load_instructions'] = [{'index':i,'instruction':line.strip()} for i,line in enumerate(instructions) if 'LDG' in line][:24]
                item['kernels'][stage] = value
            print(key,{s:{k:v[k] for k in ['instructions','registers','spill_stores','spill_loads','loads','control']} for s,v in item['kernels'].items()},flush=True)
            return key,item

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            report['variants'] = dict(pool.map(build,[(n,a) for n in sources for a in ['sm89','default']]))
    counts = {key:{stage:v['counts'] for stage,v in item.get('kernels',{}).items()} for key,item in report['variants'].items()}
    for key, item in report['variants'].items():
        arch = key.rsplit('_',1)[1]
        for stage, value in item.get('kernels',{}).items():
            own = value.pop('counts')
            base = counts['baseline_'+arch][stage]
            value['mnemonic_delta_vs_baseline'] = {k:own.get(k,0)-base.get(k,0) for k in sorted(own.keys()|base.keys()) if own.get(k,0)!=base.get(k,0)}
    (out/'summary.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    for name, source in sources.items():
        if name=='baseline':
            continue
        patch = ''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='a/candidates/pinning/pinning.cu',tofile='b/candidates/pinning/pinning.cu'))
        (out/(name+'.patch')).write_text(patch)
    print('retained_bytes',sum(f.stat().st_size for f in out.iterdir() if f.is_file()))


if __name__=='__main__':
    main()
