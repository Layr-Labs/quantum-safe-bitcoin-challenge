#!/usr/bin/env python3
"""Compose exact checked cubic recovery, repaired leaf pairs and PR137 host packing.

Only the host packing function and invocation are imported from PR137. No paired
SHA kernel or host drain changes are imported. Preserve frozen donors and output.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from preflight import source_identity
from check_candidate import function

base = ROOT/'research/formula_fusion/candidate'
leaf = ROOT/'research/leaf_pair/candidate'
base_id, leaf_id = source_identity(base), source_identity(leaf)
assert base_id['source_fingerprint'] == '7afe2d8642468c1245febc0aa9d159c76e6724e44035497a1d6623b75256c37e'
assert leaf_id['source_fingerprint'] == '94e46e120d921d87273fd9e50584cc2294ba58c57f9de6e7c8aa67b102a14b5f'
sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
public = (HERE/'pr137-window_schedule_shared.cuh').read_text()
signature = 'static int qsb_pack_second_classes('
packer = function(public, signature)
assert sha(packer) == '5b18ba41b5d44f423691b3115d2e6c0e11c75aafdc059871eca90fdf27401caa'
window_name = 'tests/gpu_epochs/window_schedule_shared.cuh'
window = (base/window_name).read_text()
for sig in ['static uint32_t qsb_window_first_key(', 'static uint32_t qsb_window_second_key(']:
    assert function(public,sig) == function(window,sig)
schedule_sig = 'static int qsb_prepare_window_schedule('
hash_sig = '__device__ __forceinline__ void qsb_scheduled_window_hash('
comment = public[public.index('/* Keep each second-block schedule class inside one warp.'):public.index(signature)]
assert signature not in window
assert window.count(schedule_sig) == 1
new_window = window.replace(schedule_sig, comment+packer+'\n\n'+schedule_sig, 1)
for sig in [schedule_sig, hash_sig]:
    assert function(new_window,sig) == function(window,sig)

tree_name = 'tests/gpu_epochs/tree.cu'
tree = (base/tree_name).read_text()
upload = '        wide_cuda_require(cudaMemcpyToSymbol(WIN3, h_win3, sizeof(h_win3)), "startup cudaMemcpyToSymbol");'
call = '        if(qsb_pack_second_classes(h_win3))return 1;\n'
assert tree.count(upload) == 1 and 'qsb_pack_second_classes(' not in tree
new_tree = tree.replace(upload,call+upload,1)
sort_end = '            memcpy(h_win3[j],w,3);\n        }'
assert new_tree.index(sort_end) < new_tree.index(call) < new_tree.index(upload)
assert new_tree.index(upload) < new_tree.index('        if (qsb_prepare_window_schedule(dp.dummy_sigs, h_win3, h_const_words)) return 1;')
assert new_tree.replace(call,'',1) == tree

dest = HERE/'candidate'
assert not dest.exists(), 'Preserve frozen combined candidate'
for name in [*base_id['source_sha256'],'COPYING']:
    target = dest/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(base/name,target)
(dest/window_name).write_text(new_window)
(dest/tree_name).write_text(new_tree)
inverse_name = 'tests/gpu_epochs/tree_inverse.cuh'
shutil.copyfile(leaf/inverse_name,dest/inverse_name)
assert 'if(n==64)__syncthreads();' in (dest/inverse_name).read_text()
candidate_id = source_identity(dest)
changed = [name for name,value in base_id['source_sha256'].items()
           if candidate_id['source_sha256'][name] != value]
assert sorted(changed) == sorted([tree_name,window_name,inverse_name])
manifest = {
    'status':'PREPARED_FOR_COMBINED_VALIDATION',
    'base_cubic_recovery_source':base_id,
    'repaired_leaf_pair_source':leaf_id,
    'candidate_source':candidate_id,
    'changed_from_cubic_base':changed,
    'only_tree_change':'Insert exact qsb_pack_second_classes(h_win3) call after host sort, before WIN3 upload and schedule preparation.',
    'window_header_change':'Exact PR137 host packer plus original comment; existing selection key, schedule and device hash bodies unchanged.',
    'inverse_header_change':'Exact frozen repaired PR138 header, including uniform n=64 CTA join.',
    'preserved_kernel_sha256':sha(function(tree,'__global__ void __launch_bounds__(256, 2) kernel_digest(')),
    'preserved_recovery_sha256':sha(function(tree,'__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(')),
    'public_donors':[
        {'pr':137,'commit':'103a6adc15391e07aac210a2653f8dfd7eb4d4c0','solver':'IvanLudvig',
         'source_header_sha256':sha(public),'actual_packer_sha256':sha(packer),
         'coauthor_if_used_before_promotion':'IvanLudvig'},
        {'pr':138,'commit':'2dc49dc4e4083050ffc34b9be61eb027eb8c291f','solver':'AbdelStark',
         'corrected_header_sha256':candidate_id['source_sha256'][inverse_name],
         'coauthor_if_used_before_promotion':'AbdelStark'},
    ],
    'existing_substantial_unpromoted_credit':['alvaroborras','MakiRH4','jacklightChen','ercumentyildirim'],
    'license':'GPL COPYING and source notices preserved; exact PR137 snapshot retained here and PR138 snapshot in leaf_pair.',
    'import_limits':'No PR137 paired-SHA or host-drain imports; no production/stage mutation or submission.',
    'evidence_scope':'Structural composition only. Parent owns full combined frontend, math, recovery, inverse, builder and native checks. Individual resource or sector-model changes are not additive speedups.',
    'gpu_executed':False,
    'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(HERE/'prepared-source.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(candidate_id['source_fingerprint'])
