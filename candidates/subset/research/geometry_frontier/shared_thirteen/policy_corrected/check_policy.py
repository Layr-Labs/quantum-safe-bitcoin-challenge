#!/usr/bin/env python3
"""Run actual startup policy with the existing fault-injected host API.

The geometry is the candidate's real header, never a stubbed offset. The old
shared13 source is an expected-failure regression; corrected source must pass
all 180 capacity/limit and 30 injected-error cases. No GPU execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from preflight import source_identity

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=HERE/'candidate')
parser.add_argument('--report', type=Path, default=HERE/'policy-host-results.json')
parser.add_argument('--expect-defect', action='store_true')
parser.add_argument('--widths', default='18,18,18,17,17,17,17,17,17,25,25,25,25',
                    help='Independent expected window widths; also supports the preserved fourteen-window control.')
args = parser.parse_args()
widths = [int(x) for x in args.widths.split(',')]
assert sum(widths)==256 and all(2<=x<=30 for x in widths)
prefixes = [sum(64 << (w-1) for w in widths[:c]) for c in range(len(widths))]
assert prefixes.count(48 << 20)==1, 'Expected geometry needs exactly one 48MiB prefix'
first_cold = prefixes.index(48 << 20)
source = args.source.resolve()
identity = source_identity(source)
legacy = ROOT/'research/asymmetric_windows/small48/check_policy.py'
text = legacy.read_text()

def once(old, new):
    global text
    assert text.count(old) == 1, old
    text = text.replace(old, new, 1)

once("base=HERE/'policy_candidate';identity=source_identity(base);h=base/'tests/gpu_epochs'",
     f"base=Path({str(source)!r});identity=source_identity(base);h=base/'tests/gpu_epochs'")
once("assert (h/'l2_policy.cuh').read_bytes()==(HERE/'l2_policy.cuh').read_bytes()",
     "assert identity == source_identity(base)")
text = text.replace('compact_build_table(&d_gtX,&d_gtY,dp.neg_r_inv);',
                    'compact_build_table(&d_gt,&unused_table_y,dp.neg_r_inv);')
text = text.replace('qsb_enable_small_l2_policy(d_gtX,gpu_index);',
                    'qsb_enable_small_l2_policy(d_gt,gpu_index);')
once('static void wide_cuda_require(int e,const char*){if(e)throw std::runtime_error("visible CUDA failure");}',
     'static void wide_cuda_require(int e,const char*why){if(e)throw std::runtime_error(why);}')
once("(HERE/'policy-host-results.json').write_text", f"Path({str(args.report.resolve())!r}).write_text")

# This is an independent expected geometry and boundary check, evaluated in
# the same host C++ translation unit as the actual policy and geometry.
geometry_main = '''int main(){
 const int expected[EXPECTED_COUNT]={EXPECTED_WIDTHS};
 assert(MIXED_CHUNKS==EXPECTED_COUNT);
 unsigned offset=0,shift=0;
 for(int c=0;c<EXPECTED_COUNT;++c){
  assert(mixed_bits(c)==expected[c]);
  assert(mixed_offset(c)==offset && mixed_shift(c)==int(shift));
  offset+=1u<<(expected[c]-1);shift+=expected[c];
 }
 assert(shift==256 && uint64_t(offset)*64==((48ull<<20)+(4ull<<30)));
 assert(uint64_t(mixed_offset(EXPECTED_FIRST_COLD))*64==HOT);
 int normal=0,fault=0;'''
geometry_main = geometry_main.replace('EXPECTED_COUNT',str(len(widths))).replace('EXPECTED_WIDTHS',','.join(map(str,widths))).replace('EXPECTED_FIRST_COLD',str(first_cold))
once('int main(){\n int normal=0,fault=0;', geometry_main)
args.report.parent.mkdir(parents=True, exist_ok=True)
failure = None
try:
    exec(compile(text, str(legacy)+'[actual-thirteen-policy]', 'exec'),
         {'__name__':'__main__', '__file__':str(legacy)})
except subprocess.CalledProcessError as exc:
    failure = exc

if args.expect_defect:
    assert failure is not None, 'Old wrong-boundary policy unexpectedly passed'
    diagnostic = failure.stderr or ''
    assert 'L2 policy must cover only the twelve small windows' in diagnostic, diagnostic
    report = {'status':'EXPECTED_STARTUP_DEFECT_REPRODUCED', **identity,
              'diagnostic':'Actual policy rejects mixed_offset(12) != 48MiB before API queries.',
              'returncode':failure.returncode, 'gpu_executed':False}
else:
    if failure:
        raise failure
    report = json.loads(args.report.read_text())
    assert report['status']=='PASS' and report['source_fingerprint']==identity['source_fingerprint']
    report['source_sha256']=identity['source_sha256']
    report['independent_geometry']=widths
    report['first_cold_window']=first_cold
    report['actual_after_build_call_verified']=True
assert source_identity(source)==identity
report['adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report['adapted_harness_sha256']=hashlib.sha256(text.encode()).hexdigest()
args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
