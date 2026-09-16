#!/usr/bin/env python3
"""Materialize the mixed64 control from the frozen wide-prefetch parent."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent))
from preflight import source_identity

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output',type=Path,default=HERE/'candidate')
args=ap.parse_args()
src=HERE.parent/'wide_windows/lookahead/prefetch'
assert source_identity(src)['source_fingerprint']=='d0174fa761b852b0f30f71246dd3e43b3e1a053dc9c011b8a4508d246eb0af18'
if args.output.exists():raise SystemExit('Refusing to overwrite an existing candidate')
shutil.copytree(src,args.output)
gpu=args.output/'tests/gpu_epochs'
p=gpu/'wide_geometry.cuh'
s=p.read_text().replace('WIDE_CHUNKS=10','WIDE_CHUNKS=15')
s=s.replace('return c<6?26:25;','return c==0?18:17;')
s=s.replace('return c<6?(unsigned)c<<25:(6u<<25)+((unsigned)(c-6)<<24);','return c==0?0u:(unsigned)(c+1)<<16;')
s=s.replace('return c<6?26*c:156+25*(c-6);','return c==0?0:17*c+1;')
s=s.replace('WIDE_TOTAL_ENTRIES=(6ull<<25)+(4ull<<24)','WIDE_TOTAL_ENTRIES=1ull<<20')
s=s.replace('(16ull<<30),"ten-window table must occupy16GiB"','(64ull<<20),"mixed fifteen-window table must occupy64MiB"')
p.write_text(s)
p=gpu/'tree.cu'
s=p.read_text().replace('#define GT_LO 8192','#define GT_LO 256').replace('#define GT_HI 8192','#define GT_HI 1024')
s=s.replace('Experimental ten-window table: [26x6,25x4], 16 GiB.','Mixed fifteen-window table: [18,17x14], 64 MiB.')
p.write_text(s)
p=gpu/'wide_table_kernels.cuh'
s=p.read_text().replace('m=2*index+1=hi*8192+lo','m=2*index+1=hi*256+lo')
s=s.replace('*ch=t<(6u<<25)?(int)(t>>25):6+(int)((t-(6u<<25))>>24);','*ch=t<(1u<<17)?0:1+(int)((t-(1u<<17))>>16);')
s=s.replace('*hi=m>>13;*lo=m&8191u;','*hi=m>>8;*lo=m&255u;')
p.write_text(s)
identity=source_identity(args.output)
assert identity['source_fingerprint']=='3205a7177cb0a51fcb269b5ea1f06daedc7ef0b6c7000029095ad86c49d1bb0b'
print(json.dumps(identity,indent=2))
