#!/usr/bin/env python3
"""Reuse the existing OpenSSL source-expression audit for both actual template modes.

The original audit supplies independent EC multiplication and recovery oracles.
Only its input selection, virtual table geometry, and template call are adapted.
This does not execute CUDA field primitives, the full GPU table, or timings.
"""
from pathlib import Path
import argparse, sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--mode',choices=('compact','wide'),required=True)
args=ap.parse_args();wide=args.mode=='wide';n=10 if wide else 15
text=(HERE.parent/'wide_windows/check_wide.py').read_text()
text=text.replace("HERE=Path(__file__).resolve().parent",'HERE=Path(__file__).resolve().parent')
start=text.index('    ap=argparse.ArgumentParser(')
end=text.index('    hashes=',start)
text=text[:start]+'''    base=HERE/'candidate'
    integrated=True
    paths=[base/name for name in ('pinning.cu','GPUMath.h','wide_geometry.cuh','compact_geometry.cuh')]
'''+text[end:]
text=text.replace('geometry=geometry_path.read_text()',"geometry=(base/'wide_geometry.cuh').read_text()+'\\n'+(base/'compact_geometry.cuh').read_text()")
text=text.replace("    wide_body=(function", "    wide_body='template<bool WIDE>\\n'+(function")
text=text.replace('wide_fixed(out,out+4,out+8,out+12,k,nullptr);',f'wide_fixed<{str(wide).lower()}>(out,out+4,out+8,out+12,k,nullptr);')
if not wide:
    # Change only the virtual loader, recoding checker and geometry oracle.
    # The actual candidate function body is extracted at runtime and unchanged.
    for old,new in [('WIDE_CHUNKS','MIXED_CHUNKS'),('wide_offset','mixed_offset'),('wide_entries','mixed_entries'),('wide_shift','mixed_shift'),('wide_bits','mixed_bits')]:
        text=text.replace(old,new)
text=text.replace('expected_bits=[26]*6+[25]*4','expected_bits='+repr([26]*6+[25]*4 if wide else [18]+[17]*14))
text=text.replace('(C.c_uint64*30)()',f'(C.c_uint64*{3*n})()').replace('(C.c_int32*10)()',f'(C.c_int32*{n})()')
text=text.replace('offset*64==16<<30',f'offset*64=={(16<<30) if wide else (64<<20)}')
text=text.replace("'source_base':args.base", "'source_base':'adaptive_geometry/candidate', 'mode':"+repr(args.mode))
text=text.replace("'geometry_cases':10,'table_bytes':16<<30",f"'geometry_cases':{n},'table_bytes':{(16<<30) if wide else (64<<20)}")
text=text.replace("{'multiplications':60,'squares':18}",repr({'multiplications':60 if wide else 95,'squares':18 if wide else 28}))
start=text.index('    filename=')
end=text.index('    result[',start)
text=text[:start]+f"    filename='cpu-{args.mode}-results.json'\n"+text[end:]
text=text.replace('No 16GiB table allocation','No full table allocation')
text=text.replace('CPU-extracted wide chain with selected base recoder and point expressions; OpenSSL field/table/oracle','CPU-extracted selected geometry chain and recovery; OpenSSL field/table/oracle')
sys.argv=[sys.argv[0]]
exec(compile(text,str(HERE/'check_geometry.py'),'exec'),{'__name__':'__main__','__file__':str(HERE/'check_geometry.py')})
