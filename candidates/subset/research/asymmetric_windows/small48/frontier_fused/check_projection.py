#!/usr/bin/env python3
"""C++ projection type check only; no CUDA execution or compilation claim."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research'))
from preflight import source_files, source_identity
from check_host_syntax import STUB

EXTRA = r'''
void __syncwarp(unsigned=0xffffffffu);
constexpr int cudaErrorNotSupported=801,cudaErrorUnsupportedLimit=215;
constexpr int cudaDevAttrMaxPersistingL2CacheSize=108,cudaDevAttrMaxAccessPolicyWindowSize=109;
constexpr int cudaLimitPersistingL2CacheSize=6,cudaStreamAttributeAccessPolicyWindow=1;
constexpr int cudaAccessPropertyPersisting=2,cudaAccessPropertyNormal=0;
struct cudaStreamAttrValue {
 struct {void *base_ptr;size_t num_bytes;float hitRatio;int hitProp,missProp;} accessPolicyWindow;
};
int cudaDeviceGetAttribute(int*,int,int);
int cudaDeviceGetLimit(size_t*,int);
int cudaStreamSetAttribute(int,int,cudaStreamAttrValue*);
'''

base = HERE / 'candidate'
identity = source_identity(base)
stripped = 0
with tempfile.TemporaryDirectory(prefix='qsb-fused-cxx-projection-') as td:
    root = Path(td)
    (root / 'cuda_runtime.h').write_text(STUB + EXTRA)
    for path in source_files(base):
        data, n = re.subn(r'<<<.*?>>>', '', path.read_text(), flags=re.S)
        stripped += n
        data = data.replace('asm volatile (', 'asm(').replace('asm volatile(', 'asm(')
        destination = root / path.relative_to(base)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(data)
    for entry in ('subset.cu', 'tests/gpu_epochs/tree_audit.cu'):
        subprocess.run(['c++', '-x', 'c++', '-std=c++17', '-fsyntax-only',
                        '-Wno-deprecated-declarations', '-DQSB_ZEROS_N=24',
                        '-I' + str(root), '-I/opt/homebrew/opt/openssl@3/include',
                        str(root / entry)], check=True)
assert identity == source_identity(base)
report = {
    'status': 'PASS',
    'validation_level': 'C++ projection syntax/types only; PTX and CUDA launch syntax removed',
    **identity,
    'launch_configurations_stripped': stripped,
    'entries': ['subset.cu', 'tests/gpu_epochs/tree_audit.cu'],
    'stub_sha256': hashlib.sha256((STUB + EXTRA).encode()).hexdigest(),
    'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'cuda_compiled': False,
    'gpu_executed': False,
}
(HERE / 'projection-results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, indent=2))
