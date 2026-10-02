from pathlib import Path
import json
p=Path('candidates/subset/center_square_sc.cuh');s=p.read_text();f=s[s.index('__device__ __forceinline__ void qsb_filter_sqr_sub2'):];f=f.replace('qsb_filter_sqr_sub2','qsb_filter_sqr_offset').replace('const uint64_t c[4], uint32_t &bad)', 'const uint64_t c[4], uint32_t high, uint32_t &bad)')
a=f.index('        "sub.cc.u32 z0,z0,u0;');b=f.index('        "\\n\\tmov.u32 %4, 0;',a)
ptx=['add.cc.u32 z0,z0,u0;']+[f'addc.cc.u32 z{i},z{i},u{i};' for i in range(1,8)]+['addc.u32 z8,z8,%17;']+['sub.cc.u32 z0,z0,v0;']+[f'subc.cc.u32 z{i},z{i},v{i};' for i in range(1,8)]+['subc.u32 z8,z8,0;', 'mov.b64 sfz,{z0,z8};','mul.wide.u32 sft,z8,977;','add.cc.u64 sft,sft,sfz;','addc.u32 fc,0,0;','mov.b64 {z0,fh},sft;','add.cc.u32 z1,z1,fh;','addc.u32 z2,z2,fc;']
f=f[:a]+'''        /* Unsigned first-fold offset (3p-c^2) minus p1. 3p rather than
         * 2p ensures nonnegativity for even noncanonical 256-bit p1.
         * Like inherited unsigned split3p-X3, top-word wrap is deliberately
         * omitted in this speculative filter. No all-domain exact claim. */
'''+''.join('        '+json.dumps(t+'\n\t')+'\n' for t in ptx)+f[b:];f=f.replace('"l"(c[2]),"l"(c[3]));','"l"(c[2]),"l"(c[3]),"r"(high));');f=f.replace('_ModSub256(r,r,(uint64_t*)b);', '_ModAdd256(r,r,(uint64_t*)b);')
# Host fallback never consumes high; reject exact route at use-site rather than
# pretending this fallback is equivalent to the uploaded 257/258-bit offset.
p=Path('candidates/subset/center_square_unsigned_sc.cuh');p.write_text('''#pragma once
/* Experimental unsigned counterpart of center_square_sc.cuh. GPL-3 inherited
 * square core. Host BN prebias is 3p-c^2; high is 1 or 2. Filter-only path. */
'''+f.rstrip()+'\n')
