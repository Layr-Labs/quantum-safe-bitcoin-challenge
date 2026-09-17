#!/usr/bin/env python3
"""Prepare an isolated corrected PR77 fused grinder with guarded small48 tables.

Both donor include closures are immutable hash anchors. Existing output is never
overwritten. This generator does not modify production or run a CUDA compiler.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from preflight import source_identity

CONTROL = ROOT / 'research/resident_projective/pr77/control'
DONOR = HERE.parent / 'frontier_schedule/candidate'
DEST = HERE / 'candidate'
CONTROL_HASH = 'e18e355f7d8c90452e6ad61a6cbf4be1a495b5bb7d0817a1e8d00a69d935c918'
DONOR_HASH = '140133ddb7a0408750a5396ce20658a1b48591422b5974315c8b76110de3117c'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def function(text, marker):
    start = text.index(marker)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def replace_region(text, start, end, replacement):
    assert text.count(start) == text.count(end) == 1, (start, end)
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement + text[finish:]


def check_standalone_calls(text, names):
    """Wrap standalone startup CUDA calls; keep existing conditional checks."""
    edits = []
    for match in re.finditer(r'\b(' + '|'.join(names) + r')\(', text):
        opening = match.end() - 1
        depth = 1
        end = opening + 1
        while depth:
            depth += (text[end] == '(') - (text[end] == ')')
            end += 1
        if text[end:].lstrip().startswith(';'):
            call = text[match.start():end]
            edits.append((match.start(), end,
                          'wide_cuda_require(' + call + ', "startup ' + match[1] + '")'))
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text, len(edits)


def main():
    assert not DEST.exists(), 'Preserve frozen source; existing candidate must not be overwritten'
    control_id = source_identity(CONTROL)
    donor_id = source_identity(DONOR)
    assert control_id['source_fingerprint'] == CONTROL_HASH, control_id
    assert donor_id['source_fingerprint'] == DONOR_HASH, donor_id
    files = {name: (CONTROL / name).read_text()
             for name in [*control_id['source_sha256'], 'COPYING']}
    prefix = 'tests/gpu_epochs/'
    for name in ['compact_geometry.cuh', 'compact_table_device.cuh',
                 'compact_table_kernels.cuh', 'compact_table_host.cuh', 'l2_policy.cuh']:
        files[prefix + name] = (DONOR / prefix / name).read_text()

    device = files[prefix + 'compact_table_device.cuh']
    old_seed = '_PointAddXYZZ_mm(X,Y,ZZ,ZZZ, x0,y0, x1,y1);'
    new_seed = '_PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ, x0,y0, x1,y1);'
    old_add = '_PointAddXYZZ<true>(X,Y,ZZ,ZZZ,cx,cy,y0);'
    new_add = '_PointAddXYZZ_def(X,Y,ZZ,ZZZ,cx,cy,y0,true);'
    assert device.count(old_seed) == device.count(old_add) == 1
    files[prefix + 'compact_table_device.cuh'] = device.replace(old_seed, new_seed).replace(old_add, new_add)

    # Only the reusable product/checkpoint/root helpers are needed for startup.
    # Their bodies are byte-identical to the checked donor; qsb_field_mul and
    # qsb_block_inverse_tree resolve to corrected PR77 in this translation unit.
    pipeline = (DONOR / prefix / 'ranked_pipeline.cuh').read_text()
    checkpoint = pipeline[:pipeline.index('/* Shared-denominator recovery directly from XYZZ coordinates.')]
    files[prefix + 'builder_checkpoint.cuh'] = (
        '// Startup-only excerpt. Search retains the fused PR77 per-CTA inverse.\n' + checkpoint)
    require = function((DONOR / prefix / 'wide_table_host.cuh').read_text(),
                       'static void wide_cuda_require(')
    files[prefix + 'startup_check.cuh'] = '#pragma once\n' + require + '\n'

    original = files[prefix + 'tree.cu']
    tree = original.replace('#include <cuda_runtime.h>',
                            '#include <cuda_runtime.h>\n#include <vector>\n#include "startup_check.cuh"')
    geometry = '''// Guarded fourteen-window fixed-base geometry. The fused PR77 search
// kernel and per-CTA inverse are preserved; only its fixed-base helper changes.
#include "compact_geometry.cuh"
#define GT_CHUNKS MIXED_CHUNKS
#define GT_TOTAL_ENTRIES MIXED_TOTAL_ENTRIES
__host__ __device__ __forceinline__ unsigned gt_entries(int c){return mixed_entries(c);}
__host__ __device__ __forceinline__ unsigned gt_offset(int c){return mixed_offset(c);}
__host__ __device__ __forceinline__ int gt_shift(int c){return mixed_shift(c);}

'''
    tree = replace_region(tree, '/* ============================================================\n * Signed-digit fixed-base geometry',
                          '/* n = secp256k1 group order', geometry)
    # Keep the exact PR77 reduction/odd-representative setup and digit mapping.
    digit = function(original, '__device__ __forceinline__ void gt_digit_idx(')
    replacement = digit + '''

#include "compact_table_device.cuh"
__device__ __forceinline__ void gt_recode_signed(const uint64_t k[4],int32_t e[GT_CHUNKS]){
    compact_recode_signed(k,e);
}
__device__ void _FixedBaseSignedXYZZStream(uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
                                         const uint64_t k[4],const uint8_t *gTable){
    compact_fixed_xyzz(X,Y,ZZ,ZZZ,k,gTable,nullptr);
}

'''
    tree = replace_region(tree, 'template<int BITS>\n', '/* DER checks */', replacement)

    tree = tree.replace('#include "tree_inverse.cuh"',
                        '#include "tree_inverse.cuh"\n#include "builder_checkpoint.cuh"')
    # Remove the old million-entry kernel, full-table host copy and unbounded
    # OpenSSL fallback. The bounded donor builder is the sole table producer.
    kernel_start = tree.index('__global__ void kernel_build_gtable(')
    comment_start = tree.rfind('/* ============================================================', 0, kernel_start)
    host_start = tree.index('/* ============================================================\n * Host code', kernel_start)
    tree = tree[:comment_start] + '#include "compact_table_kernels.cuh"\n\n' + tree[host_start:]
    tree = replace_region(tree, '/* Affine (x,y) of a point, as the 4+4 little-endian limbs the table uses. */',
                          '/* Digest params loader */',
                          '#include "compact_table_host.cuh"\n#include "l2_policy.cuh"\n\n')
    tree = replace_region(tree, '    size_t gt_sz = (size_t)GT_TOTAL_ENTRIES*64;',
                          '    /* Upload params */', '''    // Establish stack reservation before the mandatory table builder launch.
    wide_cuda_require(cudaDeviceSetLimit(cudaLimitStackSize,se_mode?4096:32768),
                      "set fused subset stack limit");
    uint8_t *d_gt=nullptr,*unused_table_y=nullptr;
    compact_build_table(&d_gt,&unused_table_y,dp.neg_r_inv);
    if(se_mode)qsb_enable_small_l2_policy(d_gt,gpu_index);

''')
    old_stack = '    cudaDeviceSetLimit(cudaLimitStackSize, 32768);\n'
    assert tree.count(old_stack) == 1
    tree = tree.replace(old_stack, '')
    # Check standalone CUDA setup calls without changing the measured search
    # loop or the already-checked schedule/constant preparation branches.
    main_start = tree.index('int main(')
    timer_start = tree.index('    struct timespec t0, t1, t_last_report;', main_start)
    host, wrapped = check_standalone_calls(tree[main_start:timer_start],
        ['cudaSetDevice', 'cudaGetDeviceProperties', 'cudaGetDeviceCount',
         'cudaMalloc', 'cudaMemcpy', 'cudaMemcpyToSymbol', 'cudaMemset'])
    allocation = '    uint8_t *h_combos = (uint8_t*)malloc(BATCH * t_sel);'
    assert host.count(allocation) == 1
    host = host.replace(allocation, allocation + '\n    if(!h_combos){fprintf(stderr,"Host combination allocation failed\\n");return 2;}')
    tree = tree[:main_start] + host + tree[timer_start:]
    files[prefix + 'tree.cu'] = tree

    preserved = {}
    for marker in ['__global__ void __launch_bounds__(256, 2) kernel_digest(',
                   '__global__ void kernel_build_epochs(',
                   '__device__ __forceinline__ void qsb_field_mul(',
                   '__device__ __forceinline__ void qsb_xyzz_finish_prepare(',
                   '__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(',
                   '__device__ __forceinline__ void gt_recode_setup(']:
        before = function(original, marker)
        assert function(tree, marker) == before, marker
        preserved[marker] = digest(before)
    search_marker = '    /* Short-epoch path: producer/consumer on GPU, one 256-thread block per'
    assert tree[tree.index(search_marker):] == original[original.index(search_marker):]
    assert 'ranked_pipeline.cuh' not in tree
    assert 'qsb_launch_ranked_pipeline' not in tree
    assert 'd_pipe_' not in tree
    assert 'compute_gtable' not in tree and 'chk_table' not in tree
    for name, contents in files.items():
        destination = DEST / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents)
    identity = source_identity(DEST)
    assert len(identity['source_sha256']) == len(files) - 1
    receipt = {
        'control_source': control_id,
        'geometry_builder_policy_donor': donor_id,
        'candidate_source': identity,
        'preserved_fused_functions_sha256': preserved,
        'preserved_search_tail_sha256': digest(tree[tree.index(search_marker):]),
        'changed_functions': {
            '_FixedBaseSignedXYZZStream': 'Calls guarded compact_fixed_xyzz: cold windows12,13 then small windows0..11.',
            'compact_fixed_xyzz': 'Donor body with exactly two equivalent PR77 deferred-add symbol adaptations; final guard unchanged.',
            'gt_recode_signed': 'Delegates to exact donor geometry recoder; PR77 gt_recode_setup is unchanged.',
            'main_startup': 'Checked device setup/allocations/copies; early4096-byte ranked stack; bounded mandatory table builder; optional checked48MiB L2 priority.',
        },
        'builder_dependencies': {
            'builder_checkpoint.cuh': 'Exact donor prefix through qsb_root_group_finish; no ranked prepare, finish or search-launch functions.',
            'qsb_field_mul': 'Exact correctedPR77 function; different source from external donor, requires integrated builder verification.',
            'qsb_block_inverse_tree': 'Exact correctedPR77 per-CTA tree; also serves builder super-root inversion.',
            'GPUMath.h': 'Byte-identical correctedPR77; donor builder invokes these field primitives.',
        },
        'startup_resources': {
            'table_bytes': ((48 << 20) + (4 << 30)),
            'table_entries': ((12 << 16) + (2 << 25)),
            'host_ladder_array_bytes': 2 * 14 * 8192 * 64,
            'host_ec_points_per_runtime_base': 73906,
            'gpu_builder_chunk_entries': 1 << 20,
            'gpu_builder_batches': 65,
            'gpu_builder_temporary_bytes': 2 * 14 * 8192 * 64 + 4096 * 32 + 4096 * 4 * 256 * 8 + 16 * 32 + 16 * 4 * 256 * 8,
            'sampled_table_readback_bytes': (14 * 8 + 256) * 64,
            'ranked_stack_request_bytes_per_thread': 4096,
            'external_search_checkpoint_bytes': 0,
            'hot_path_operations_normal_chain': {'multiplications': 88, 'squares': 26},
            'checked_standalone_startup_cuda_calls': wrapped,
            'whole_table_host_fallback_or_readback': False,
            'actual_GPU_startup_or_resource_measurement': False,
        },
        'unmodified_control_files': [name for name in control_id['source_sha256']
                                     if files[name] == (CONTROL / name).read_text()],
        'exact_donor_files': [name for name in files
                             if name in donor_id['source_sha256'] and files[name] == (DONOR / name).read_text()],
        'correctness_and_performance_limits': [
            'Generator structural assertions only; parent owns arithmetic, integrated builder, host API and native CUDA validation.',
            'PR77 fused kernel uses its original per-CTA inverse; the cold-first helper changes register lifetimes and may spill.',
            'Full4GiB table construction and driver-JIT startup count against the official wall clock.',
            'Optional48MiB cache policy gives priority, not guaranteed residency; unsupported policy keeps the same table geometry.',
            'No GPU executed and no score or additive speedup inferred from PR77 or the external-pipeline candidate.',
        ],
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'gpu_executed': False,
        'production_changed': False,
    }
    (HERE / 'prepared-source.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'PREPARED', **identity}, indent=2))


if __name__ == '__main__':
    main()
