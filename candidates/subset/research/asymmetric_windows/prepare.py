#!/usr/bin/env python3
"""Prepare the isolated 14-term, 52 MiB + 3 GiB table experiment."""
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from preflight import source_identity

parent = HERE.parent / 'startup_budget/candidate'
receipt = json.loads((HERE.parent / 'startup_budget/prepared-source.json').read_text())
identity = source_identity(parent)
for name, digest in identity['source_sha256'].items():
    assert receipt['candidate_source_sha256'][name] == digest, name
production = source_identity(ROOT)
assert production['source_fingerprint'] == '6bfe0e61fbbcabce237877c95e238f0bd22914ed6ceade6baf97e058c5e51ac6'
dest = HERE / 'candidate'
for name in [*identity['source_sha256'], 'COPYING']:
    p = dest / name
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(parent / name, p)
h = dest / 'tests/gpu_epochs'
p = h / 'compact_geometry.cuh'
s = p.read_text()
begin = s.index('constexpr int MIXED_CHUNKS=')
end = s.index('// Input M is odd.')
s = s[:begin] + '''// Twelve small windows cover205 bits; two cold windows cover the final51.
// The compact_ namespace is retained to isolate geometry from pipeline changes.
constexpr int MIXED_CHUNKS=14;
__host__ __device__ __forceinline__ int mixed_bits(int c){return c==0?18:(c<12?17:(c==12?26:25));}
__host__ __device__ __forceinline__ unsigned mixed_entries(int c){return 1u<<(mixed_bits(c)-1);}
__host__ __device__ __forceinline__ unsigned mixed_offset(int c){
    return c==0?0u:(c<13?((unsigned)(c+1)<<16):((13u<<16)+(1u<<25)));
}
__host__ __device__ __forceinline__ int mixed_shift(int c){return c==0?0:(c<13?17*c+1:231);}
constexpr uint64_t MIXED_TOTAL_ENTRIES=(13ull<<16)+(1ull<<25)+(1ull<<24);
static_assert(MIXED_TOTAL_ENTRIES*64==((52ull<<20)+(3ull<<30)),"asymmetric table size");
''' + s[end:]
p.write_text(s.replace('Research-only mixed signed-window geometry.', 'Research-only asymmetric signed-window geometry.'))
p = h / 'compact_table_device.cuh'
s = p.read_text().replace('#define COMPACT_LO 256', '#define COMPACT_LO 8192').replace('#define COMPACT_HI 1024', '#define COMPACT_HI 8192')
p.write_text(s)
p = h / 'compact_table_kernels.cuh'
s = p.read_text().replace('hi*256+lo', 'hi*8192+lo')
s = s.replace('*ch=t<(1u<<17)?0:1+(int)((t-(1u<<17))>>16);', '*ch=t<(1u<<17)?0:(t<(13u<<16)?1+(int)((t-(1u<<17))>>16):(t<((13u<<16)+(1u<<25))?12:13));')
s = s.replace('*hi=m>>8;*lo=m&255u;', '*hi=m>>13;*lo=m&8191u;')
p.write_text(s)
p = h / 'compact_table_host.cuh'
p.write_text(p.read_text().replace('allocate compact64MiB table', 'allocate asymmetric14 table').replace('Wide table', 'Asymmetric table'))
p = h / 'tree.cu'
s = p.read_text()
needle = 'd_wide=qsb_optional_wide_table(se_mode,dp.neg_r_inv);'
assert s.count(needle) == 1
s = s.replace(needle, 'd_wide=nullptr; // Isolated asymmetric14 experiment; no optional16GiB table.')
s = s.replace('// The compact64MiB path is always available. Generic modes use it.', '// Isolated asymmetric14 path; ranked and generic modes use the same geometry.')
p.write_text(s)
widths = [18] + [17]*11 + [26, 25]
report = {
    'inherited_source': identity, 'candidate_source': source_identity(dest),
    'geometry_bits': widths, 'table_entries': sum(1 << (b-1) for b in widths),
    'table_bytes': sum(64 << (b-1) for b in widths),
    'small_region_bytes': 52 << 20, 'cold_region_bytes': 3 << 30,
    'chain_operations': {'multiplications': 88, 'squares': 26, 'point_lookups': 14},
    'full_ladder_points_per_base': sum(4096+(1 << (b-13))-1 for b in widths),
    'changes': 'Geometry/decoder/ladders only; optional16GiB allocation disabled. Inherits checked startup. Field, hash, inverse and deferred-Y formulas unchanged.',
    'gpu_executed': False,
    'limits': 'No cache-residency or throughput claim. Retained compact_ symbols name the inherited dispatch slot, not a64MiB table.',
}
assert source_identity(ROOT) == production
(HERE / 'prepared-source.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if not k.endswith('source')}, indent=2))
