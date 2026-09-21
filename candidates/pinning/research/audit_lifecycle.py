from pathlib import Path
import random,json,re,hashlib
w=Path(__file__).resolve().parent;p=w.parent;s=(p/'pinning.cu').read_text();h=(p/'IdentityFlags.cuh').read_text();host=(p/'Serial16Host.h').read_text();r=random.Random(1758)
# Extract original launch helper, which remains the stream-ordering boundary.
a=s.index('static void launch_pinning_pipeline(');b=s.index('\n}',a)+2;launch=s[a:b]
assert launch.index('kernel_pinning_pipeline<FAST_TAIL,0>')<launch.index('qsb_root_group_prepare<<<')<launch.index('qsb_invert_super_roots<<<')<launch.index('qsb_root_group_finish<<<')<launch.index('kernel_pinning_pipeline<FAST_TAIL,2>')
assert launch.count('saved,roots,tree,tp);')==2
assert launch.count('QSB_STREAM_ARG')==7 # includes two compile-disabled offload launch sites
assert '#if QSB_TREE_OFFLOAD || QSB_TREE_OFFLOAD2' in s
assert s.index('if (drain_slot(s)) return 1;')<s.index('d_pipeline_state[s],d_pipeline_roots[s],d_pipeline_tree[s],')
assert 'cudaStream_t st = slot_stream[s];' in s
assert '__ballot_sync(0xffffffffu,idx<n && special==1u)' in h
assert '(threadIdx.x&31u)==0u && idx<n' in h
# Source-bound dispatch prevents cold return before ballot; whole inactive blocks return uniformly.
a=s.index('__device__ void _FixedBaseSignedXYZZScalar(');b=s.index('\n}',a);f=s[a:b]
assert f.index('qsb_s16_publish_identity(')<f.index('if(special)')
assert 'if (blockIdx.x * blockDim.x >= batch_size) return;' in s
assert 'int active = idx < batch_size;' in s
# Confirm flags are consulted only for zero saved-V, never used to interpret arbitrary field values.
assert 'if(identity && !qsb_s16_identity_flag((const uint32_t*)tree,(unsigned)idx))return;' in s
assert 'identity ? qsb_serial16_identity_finish(u2rx,u2ry,q1x,q2x)' in s
for token in ['BN_set_word(k,65535)','BN_lshift(k,k,240)','BN_mod_mul(k,k,nri,order,ctx)','BN_lebin2bn(nri_bytes,32,nri)','BN_bn2lebinpad(x,bytes,32)','BN_bn2lebinpad(ny,bytes+64,32)']:
 assert token in host
assert 'cudaMemcpyToSymbol(qsb_serial16_special_xy,special_xy,sizeof(special_xy))!=cudaSuccess' in s
# Interleave different streams arbitrarily, preserving per-stream FIFO. Include varying tail sizes
# and reuse: no word left from an earlier larger batch may affect a later smaller one.
rounds=0;checked=0
for slots in [1,2,3]:
 for width in [64,128,256]:
  storage=[[0xdeadbeef]*129 for _ in range(slots)]
  for generation in range(64):
   tasks=[];expected={};ns={}
   for slot in range(slots):
    n=r.randrange(1,4097);ns[slot]=n
    marks={i for i in range(n) if r.randrange(53)==0};expected[slot]=marks
    tasks.append([slot,'prepare']);tasks.append([slot,'root']);tasks.append([slot,'finish'])
   pending={slot:['prepare','root','finish'] for slot in range(slots)}
   while any(pending.values()):
    slot=r.choice([j for j in pending if pending[j]]);stage=pending[slot].pop(0);n=ns[slot]
    if stage=='prepare':
     for base in range(0,((n+width-1)//width)*width,32):
      bits=sum((base+j<n and base+j in expected[slot])<<j for j in range(32))
      if base<n:storage[slot][base>>5]=bits
    elif stage=='finish':
     found={i for i in range(n) if storage[slot][i>>5]>>(i&31)&1};assert found==expected[slot];checked+=n
   rounds+=1
result={'status':'PASS','stream_generations':rounds,'active_candidate_flag_checks':checked,'slots':[1,2,3],'prepare_widths':[64,128,256],'host_scalar_formula':'65535*2^240*neg_r_inv mod n','scope':'Static callsite/order audit plus model of CUDA FIFO/slot reuse. Not native compilation or a GPU race detector. Other inherited P=+/-R recovery skips remain unchanged.'}
(w/'lifecycle-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
