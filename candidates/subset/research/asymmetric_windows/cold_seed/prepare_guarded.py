#!/usr/bin/env python3
"""Preserve the unguarded control; handle its final equal/opposite-point cases."""
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from preflight import source_identity

parent=HERE/'candidate'
identity=source_identity(parent)
assert identity['source_fingerprint']=='c61bb198f730bc94fef3ceb0c4fddbba09b544175e9e93a284fd38798c47bd5f'
dest=HERE/'guarded_candidate'
assert not dest.exists(),'Preserve the existing generated source'
for name in [*identity['source_sha256'],'COPYING']:
    out=dest/name;out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(parent/name,out)

math=(parent/'GPUMath.h').read_text()
start=math.index('template<bool DEFER_Y>')
end=math.index('\n// Direct-three-affine prefix',start)
helper=math[start:end].replace('template<bool DEFER_Y>\n','',1)
helper=helper.replace('_PointAddXYZZ(', 'qsb_asym_last_add(',1)
helper=helper.replace('if (DEFER_Y)', 'if (false)')
marker='  _ModSqr(PP, P);'
assert helper.count(marker)==1
guard=r'''
  // The cold-first ordering has no exceptional prefix before this final add.
  // A zero x difference is either doubling (R=0) or opposite points (R!=0).
  // Construct 2*(X2,Y2) directly from the affine table point in the rare case.
  if (!(P[0]|P[1]|P[2]|P[3])) {
    if (R[0]|R[1]|R[2]|R[3]) {
      for(int i=0;i<4;++i) X1[i]=Y1[i]=ZZ1[i]=ZZZ1[i]=0;
      return;
    }
    uint64_t xx[4],yy[4],yyyy[4],s[4],m[4],t[4],tmp[4];
    _ModSqr(xx,(uint64_t *)X2);
    _ModSqr(yy,(uint64_t *)Y2);
    _ModSqr(yyyy,yy);
    _ModMult(s,(uint64_t *)X2,yy);
    _ModAdd256(s,s,s);_ModAdd256(s,s,s); // s=4*x*y^2
    _ModAdd256(m,xx,xx);_ModAdd256(m,m,xx); // m=3*x^2
    _ModSqr(t,m);_ModSub256(t,t,s);_ModSub256(t,t,s);
    _ModSub256(tmp,s,t);_ModMult(Y1,m,tmp);
    _ModAdd256(yyyy,yyyy,yyyy);_ModAdd256(yyyy,yyyy,yyyy);_ModAdd256(yyyy,yyyy,yyyy);
    _ModSub256(Y1,Y1,yyyy);Load256(X1,t);
    _ModAdd256(ZZ1,yy,yy);_ModAdd256(ZZ1,ZZ1,ZZ1);
    _ModMult(ZZZ1,(uint64_t *)Y2,yy);
    _ModAdd256(ZZZ1,ZZZ1,ZZZ1);_ModAdd256(ZZZ1,ZZZ1,ZZZ1);_ModAdd256(ZZZ1,ZZZ1,ZZZ1);
    return;
  }
'''
helper=helper.replace(marker,guard+marker)
p=dest/'tests/gpu_epochs/compact_table_device.cuh';s=p.read_text()
s=s.replace('__device__ void compact_fixed_xyzz(',helper+'\n__device__ void compact_fixed_xyzz(',1)
marker='_PointAddXYZZ<false>(X,Y,ZZ,ZZZ,cx,cy,y0);'
assert s.count(marker)==1
s=s.replace(marker,'qsb_asym_last_add(X,Y,ZZ,ZZZ,cx,cy,y0);')
p.write_text(s)
report={'inherited_source':identity,'candidate_source':source_identity(dest),
 'change':'Final addition detects zero x difference; doubles its affine input when equal, represents infinity when opposite. Normal final-add field work unchanged.',
 'purpose':'Fix the independently derived last-add doubling witness, not a standalone performance improvement.',
 'gpu_executed':False}
(HERE/'guarded-source.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['candidate_source']['source_fingerprint'])
