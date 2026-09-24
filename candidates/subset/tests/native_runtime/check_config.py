#!/usr/bin/env python3
"""Preprocess the literal carrier configuration guard; no candidate build/GPU run."""
from pathlib import Path
import hashlib,json,os,re,subprocess,tempfile
HERE=Path(__file__).resolve().parent
source=HERE.parents[1]/'tests/gpu_epochs/tree.cu'
text=source.read_text()
start=text.index('#if QSB_HOST_CARRIER',text.index('// The immutable native payload implements exactly these host-visible defaults.'))
end=text.index('\nextern "C"',start)
guard=text[start:end]
expected={'ZLAB_T14':0,'QSB_SE_WINDOWS':128,'QSB_DIGIT_SHIFT':1,'ZLAB_DIRDIG':1,
          'QSB_PAIR_SHARED':1,'QSB_EPOCH_GROUPS':1,'QSB_EPOCH_FAST':1,'ZLAB_HITPATH':1,
          'ZLAB_TRIM':1,'ZLAB_PAIRSHA':0,'ZLAB_DUAL_EPOCH_SHA':1,'QSB_SE_BLOCK':256,
          'QSB_SE_EARLY':6,'QSB_SE_TWIN':3,'QSB_SE_CUT':137,'QSB_PREFIX_BLOCKS':2,
          'ZLAB_LAUNCH_BLOCKS':262144,'QSB_SHA_UNROLL_CONST':1}
assert dict((k,int(v)) for k,v in re.findall(r'(\w+)\s*!=\s*(\d+)',guard))==expected
prefix=''.join('#ifndef '+k+'\n#define '+k+' '+str(v)+'\n#endif\n' for k,v in expected.items())
with tempfile.TemporaryDirectory(prefix='carrier-config-') as td:
    path=Path(td)/'guard.cc';path.write_text(prefix+guard)
    def check(role,extra,ok):
        p=subprocess.run([os.environ.get('CXX','c++'),'-E','-P','-x','c++','-DQSB_HOST_CARRIER='+str(role)]+extra+[str(path)],text=True,capture_output=True)
        assert (p.returncode==0)==ok,(role,extra,p.stderr)
        if not ok:assert 'Native carrier requires the immutable native-v1 host geometry and mapping defaults' in p.stderr
    check(1,[],True)
    for name,value in expected.items():
        flag='-D'+name+'='+str(value+1)
        check(1,[flag],False)
        check(0,[flag],True)
print(json.dumps({'status':'PASS','gpu_executed':False,'candidate_compiled':False,
                 'carrier_baseline_accepted':True,'carrier_incompatible_configurations_rejected':len(expected),
                 'full_source_guard_bypasses':len(expected),'guarded_defaults':expected,
                 'tree_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                 'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
