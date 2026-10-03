#!/usr/bin/env python3
"""Integrity audit of diagnostics; NOT a new cryptographic/performance verifier."""
from pathlib import Path
import io,json,tarfile,lzma,hashlib,base64,re,sys,csv
from iter49_csv_codec import decode
from iter47b_sass_codec import reconstruct as sass
from iter49_addr_codec import transform as addr
from iter49_endian_codec import transform as endian
lab=Path(__file__).resolve().parent;base=lab.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.load(open(lab/'iter49-joint-history.tar.xz.json'));blob=(lab/'iter49-joint-history.tar.xz').read_bytes()
assert len(blob)==r['bytes'] and sha(blob)==r['sha256']
raw=endian(addr(lzma.decompress(blob),True));assert len(raw)==r['raw_tar_bytes'] and sha(raw)==r['raw_tar_sha256']
old=json.load(open(lab/'iter49-prior-history-reconstruction.json'));assert all(r['records'][k]==v for k,v in old['records'].items())
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
 assert set(t.getnames())==set(r['records'])
 for n,v in r['records'].items():
  b=t.extractfile(n).read();assert len(b)==v['bytes'] and sha(b)==v['sha256'],n
 diag=sass(t.extractfile('iter49-evidence.rawtar.normalized.sass').read())
with tarfile.open(fileobj=io.BytesIO(diag)) as t:
 s=json.load(t.extractfile('diagnostic/score.json'));run=json.load(t.extractfile('diagnostic/run.json'))
 assert s['metrics']['verified'] and s['metrics']['verified_hits']==4210 and len(run['hits'])==4210
 csvbytes=decode(t.extractfile('diagnostic/phases.csv.codec').read());rows=list(csv.DictReader(io.StringIO(csvbytes.decode())))
 assert len(rows)==56832;keys=['entry','sha','front_a','front_b','cross','inverse','tail'];groups={}
 for row in rows:
  a=[int(row[k]) for k in keys];assert all(a[i+1]>a[i] for i in range(6))
  b=int(row['batch']);assert int(row['slot'])==b%2;groups.setdefault(b,[]).append(int(row['sample']))
 assert set(groups)==set(range(111)) and all(sorted(v)==list(range(512)) for v in groups.values())
 for arch,expected in [('89','a724399a04db515f8bebd836b1af866e6e7a1b778a98ce73a384c6f7704de147'),('86','95598e445005e424dbee6feb03f87adeb4c498afe1b07250fa0e169e9edbbfa5')]:assert sha(t.extractfile('diagnostic/off'+arch+'.cubin').read())==expected
 f=json.load(t.extractfile('final/iter49-findings.json'));assert f==json.load(open(lab/'iter49-findings.json'))
assert re.search(r'#ifndef QSB_DIGEST_PHASE_AUDIT\s+#define QSB_DIGEST_PHASE_AUDIT 0',(base/'tests/gpu_epochs/digest_phase_audit.cuh').read_text())
assert not any(p.is_symlink() for p in base.rglob('*'))
with tarfile.open(lab/'iter48-cold-lab.tar.xz') as t:
 cold=json.load(t.extractfile('manifest.json'));assert set(t.getnames())==set(cold)|{'manifest.json'}
 for n,v in cold.items():
  b=t.extractfile(n).read();assert len(b)==v['bytes'] and sha(b)==v['sha256'],n

out={'existing_verifier_PASS_hits':4210,'complete_timestamp_rows_PASS':56832,'all_prior_member_hashes_PASS':True,'off_native_local_byte_identity_PASS':True,'diagnostic_not_qualified':True,'submission':None,'package_bytes':0,'limit_bytes':8388608}
p=lab/'iter49-final-audit.json'
for _ in range(10):
 p.write_text(json.dumps(out,indent=2)+'\n');n=sum(p.stat().st_size for p in base.rglob('*') if p.is_file());assert n<=8388608,n
 if out['package_bytes']==n:break
 out['package_bytes']=n
print(json.dumps(out,indent=2))
