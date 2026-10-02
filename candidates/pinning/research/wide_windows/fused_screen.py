#!/usr/bin/env python3
"""Generate a compile-only fused search variant from the integrated source.

Preserves the exact prepare and finish expressions and inherited whole-CTA
inverse. No production launch changes, GPU execution, or performance claim.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ptx_field_model import function


def make_fused(src):
    old = function(src, '__global__ void __launch_bounds__(256, STAGE == 0 ? 2 : 3) kernel_pinning_pipeline(')
    split = old.index('    if (STAGE==0) {')
    stores = old.index('    if(active){\n        /* Eight vector planes')
    finish = old.index('    uint64_t u2rx[4]')
    assert old.endswith('    }\n    }\n}')
    fused = old[:split] + old[split + len('    if (STAGE==0) {\n'):stores]
    fused += '    qsb_block_inverse(prod);\n    if (!usable) return;\n'
    fused += old[finish:-len('    }\n}')] + '}'
    fused = fused.replace('STAGE == 0 ? 2 : 3', '2').replace('kernel_pinning_pipeline(', 'kernel_pinning_fused(')
    assert 'STAGE' not in fused and 'saved[' not in fused and 'checkpoint(' not in fused
    # Both the early block return and the per-lane return retain collective
    # safety: only the latter depends on lane state and it follows the inverse.
    assert fused.index('qsb_block_inverse(prod);') < fused.index('if (!usable) return;')
    return 'template<bool FAST_TAIL>\n' + fused + '\n'


def main():
    src = (HERE / 'candidate/pinning.cu').read_text()
    fused = make_fused(src)
    args = '''const uint32_t *mid, const uint8_t *suffix, int suffix_len,
    int seq_offset, int lt_offset, int total_len, uint32_t seq, uint32_t lt,
    const uint64_t *nri, const uint64_t *rx, const uint64_t *ry,
    const uint64_t *n2x, const uint64_t *n2y, const uint8_t *table,
    uint32_t *hit_count, uint32_t *hits, int count, int easy, int single,
    uint32_t slot'''
    actual = 'mid,suffix,suffix_len,seq_offset,lt_offset,total_len,seq,lt,nri,rx,ry,n2x,n2y,table,hit_count,hits,count,easy,single,slot,nullptr,nullptr,nullptr'
    out = '#define main integrated_candidate_main\n#include "candidate/pinning.cu"\n#undef main\n\n'
    # An integrated candidate may already include this exact definition.
    if '#include "wide_fused_kernel.cuh"' not in src:
        out += fused + '\n'
    for fast in ('true', 'false'):
        out += f'void instantiate_fused_{fast}({args}) {{\n'
        out += f'    kernel_pinning_fused<{fast}><<<(count+255)/256,256>>>({actual});\n}}\n'
    out += '\nint main(){return 0;} // Compile-only probe: never launches a GPU kernel.\n'
    (HERE / 'fused_probe.cu').write_text(out)
    provenance = {'source_sha256': hashlib.sha256(src.encode()).hexdigest(),
                  'probe_sha256': hashlib.sha256(out.encode()).hexdigest(),
                  'change': 'Keep four recovery fields in registers across the original CTA product-tree inverse; remove search state and tree checkpoints.',
                  'tradeoff': 'One root inversion per256 candidates instead of one per16M; shared barriers and live registers may dominate.',
                  'gpu_executed': False}
    (HERE / 'fused-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
