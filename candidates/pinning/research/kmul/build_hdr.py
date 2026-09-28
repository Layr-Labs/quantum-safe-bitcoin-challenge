#!/usr/bin/env python3
"""Rewrite the asm body of _ModMultCoreK in ../../KaratsubaMul.cuh: the Karatsuba product
stage from gen_kmul.py followed by the _ModMultCore (QSB_SHORT_CARRY, QSB_CARRY_GLUE)
reduction tail, copied verbatim from ../../GPUMath.h."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
TRACK = HERE.parent.parent
sys.path.insert(0, str(HERE))
from gen_kmul import product_stage

g = (TRACK / 'GPUMath.h').read_text()
i = g.index('#if QSB_SHORT_CARRY\n__device__ __forceinline__ void _ModMultCore(')
k = g.index('asm("', g.index('#if QSB_CARRY_GLUE', i))
glue = g[k:g.index('\n#else', k)]
tail = glue[glue.index('QZ_ADD("x15") "\\n\\t.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;'):]

prod = ''.join('\n\t%s' % l for l in product_stage())
body = '"{" QZ_DECL "' + prod.replace('\n', '\\n').replace('\t', '\\t') + '" ' + tail
hp = TRACK / 'KaratsubaMul.cuh'
h = hp.read_text()
a = h.index('    asm(') + len('    asm(')
b = h.index('\n        : "=l"(r0)', a)
hp.write_text(h[:a] + body + h[b:])
print('rewrote', hp.name)
