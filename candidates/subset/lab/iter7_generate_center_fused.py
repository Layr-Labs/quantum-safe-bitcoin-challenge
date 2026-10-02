from pathlib import Path
import re,json
src=Path('candidates/subset/hit_filter_field_sc.cuh').read_text()
f=src[src.index('__device__ __forceinline__ void qsb_filter_sqr('):src.index('__device__ __forceinline__ void qsb_filter_mul(uint64_t*r')]
f=f.replace('void qsb_filter_sqr(uint64_t r[4], const uint64_t a[4], uint32_t &bad)', 'void qsb_filter_sqr_sub2(uint64_t r[4], const uint64_t a[4], const uint64_t b[4], const uint64_t c[4], uint32_t &bad)')
# Extra inputs are consumed only after the first fold. Carry their signed high
# word into the second fold, following the inherited signed-X3 reduction.
start=f.index('        "{ .reg .u64 sfz,sft;')
end=f.index('        "\\n\\tmov.u32 %4, 0;',start)
ptx=['.reg .u32 u0,u1,u2,u3,u4,u5,u6,u7,v0,v1,v2,v3,v4,v5,v6,v7,ext,fc,fq,fh;', '.reg .u64 sfz,sft;']
for prefix,offset in [('u',9),('v',13)]:
 for i in range(4): ptx += [f'mov.b64 {{{prefix}{2*i},{prefix}{2*i+1}}}, %{offset+i};']
for prefix in ['u','v']:
 ptx += [f'sub.cc.u32 z0,z0,{prefix}0;']
 ptx += [f'subc.cc.u32 z{i},z{i},{prefix}{i};' for i in range(1,8)]
 ptx += ['subc.cc.u32 z8,z8,0;',f'subc.u32 z9,{"0" if prefix=="u" else "z9"},0;']
ptx+=['mad.lo.u32 fq,z9,977,z8;', 'mov.b64 sfz,{z0,fq};', 'mul.wide.u32 sft,z8,977;', 'add.cc.u64 sft,sft,sfz;', 'addc.u32 fc,z9,0;', 'mov.b64 {z0,fh},sft;', 'shr.s32 ext,fc,31;', 'add.cc.u32 z1,z1,fh;', 'addc.cc.u32 z2,z2,fc;', 'addc.u32 z3,z3,ext;']
replacement='        /* First-fold square minus two 256-bit residues; signed high fold.\n         * Like inherited fused-X3, short final carry is speculative: publication\n         * still requires the existing exact host/OpenSSL verification. */\n'+''.join('        '+json.dumps(p+'\n\t')+'\n' for p in ptx)
f=f[:start]+replacement+f[end:]
f=f.replace('"l"(a[2]), "l"(a[3]));','"l"(a[2]), "l"(a[3]),\n          "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]),\n          "l"(c[0]),"l"(c[1]),"l"(c[2]),"l"(c[3]));')
f=f.replace('_ModSqr(r,a);','_ModSqr(r,a); _ModSub256(r,r,(uint64_t*)b); _ModSub256(r,r,(uint64_t*)c);')
Path('candidates/subset/center_square_sc.cuh').write_text('''#pragma once
/* Default-off centered-square offset fusion. Generated from inherited
 * hit_filter_field_sc.cuh's square core (GPL-3); generation recipe in lab.
 * The unchanged first fold has the inherited rare-carry approximations.
 * This is a speculative hit filter, NOT a proof of exact field arithmetic. */
'''+f)
