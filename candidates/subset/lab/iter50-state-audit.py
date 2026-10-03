#!/usr/bin/env python3
"""Package/evidence integrity only; benchmark.sh owns cryptographic verification."""
from pathlib import Path
import base64, hashlib, io, json, lzma, re, sys, tarfile
from iter47_normalize_carriers import reconstruct
from iter49_addr_codec import transform as addr
from iter49_endian_codec import transform as endian

lab = Path(__file__).resolve().parent
base = lab.parent
sha = lambda b: hashlib.sha256(b).hexdigest()

def unpack(b):
    with tarfile.open(fileobj=io.BytesIO(b)) as t:
        return {m.name: t.extractfile(m).read() for m in t.getmembers()}

def check(records, blobs):
    for n, r in records.items():
        assert len(blobs[n]) == r['bytes'] and sha(blobs[n]) == r['sha256'], n

r = json.loads((lab/'iter50-joint-history.tar.xz.json').read_text())
b = (lab/'iter50-joint-history.tar.xz').read_bytes()
assert len(b) == r['bytes'] and sha(b) == r['sha256']
raw = lzma.decompress(b)
assert len(raw) == r['rawbytes'] and sha(raw) == r['rawsha256']
outer = unpack(raw)
assert set(outer) == set(r['records'])
check(r['records'], outer)
prior = endian(addr(outer['prior49.rawtar'], True))
p = json.loads((lab/'iter49-joint-history.tar.xz.json').read_text())
assert len(prior) == p['raw_tar_bytes'] and sha(prior) == p['raw_tar_sha256']
prior_blobs = unpack(prior)
assert set(prior_blobs) == set(p['records'])
check(p['records'], prior_blobs)
screens_raw = reconstruct(outer['screens50.rawtar.normalized'])
assert sha(screens_raw) == r['source_screens_sha256']
screens = unpack(screens_raw)
check(json.loads(screens['manifest.json']), screens)
qualification = unpack(outer['qualification50.rawtar'])
check(json.loads(qualification['records.json']), qualification)
counts = {}
for name, blobs in [('screens', screens), ('qualification', qualification)]:
    scores = [(n, json.loads(b)) for n, b in blobs.items() if n.endswith('.score.json')]
    for n, s in scores:
        run = json.loads(blobs[n.replace('.score.json', '.run.json')])
        assert s['metrics']['verified']
        assert len(run['hits']) == run['verified_hits'] == s['metrics']['verified_hits']
        assert s['score'] == round(s['metrics']['throughput_Mps'] * 1e6)
    counts[name] = {'runs': len(scores), 'verified_hits': sum(s['metrics']['verified_hits'] for _, s in scores)}
matched = json.loads(qualification['matched-findings.json'])
assert matched['gate'] and matched['hits'] == 47123
assert all(p['vsleader_percent'] >= 4 and p['incrementality_percent'] > 0 for p in matched['pairs'])
assert json.loads(qualification['matched-manifest.json'])['compiler_route'] == 'matched native sm86 -DQSB_CARRIER_BUILD=1'
fixed = json.loads(qualification['enabledfinal.score.json'])
assert fixed['metrics']['verified'] and fixed['metrics']['verified_hits'] == 5319
with tarfile.open(lab/'iter48-cold-lab.tar.xz') as t:
    cold = {m.name: t.extractfile(m).read() for m in t.getmembers()}
check(json.loads(cold['manifest.json']), cold)
assert not any(p.is_symlink() for p in base.rglob('*'))
source = (base/'subset.cu').read_text()
assert re.search(r'#ifndef QSB_FIRST_DEVICE\s+#define QSB_FIRST_DEVICE 1', source)
assert re.search(r'#ifndef QSB_FIRST_HOSTLESS\s+#define QSB_FIRST_HOSTLESS 1', source)
assert re.search(r'#ifndef QSB_FIRST_LOCAL\s+#define QSB_FIRST_LOCAL 0', (base/'tests/gpu_epochs/window_schedule_shared.cuh').read_text())
header = (base/'qsb_carrier_sm89.h').read_text()
assert '8105a04f70973f70c62334b83ff5977039c3c4e72022467090e13ef110d04a0e' in header
assert len((lab/'ITER50-SUBMISSION.md').read_bytes()) >= 5120
assert not re.search(r'/(?:home|work)/', (lab/'ITER50-SUBMISSION.md').read_text())
out = {'matched_gate_PASS': True, 'matched_exact_verifier_hits': 47123, 'fixed_enabled_exact_verifier_hits': 5319,
       'prior49_member_hashes_PASS': True, 'screen_qualification_cold_hashes_PASS': True,
       'counts': counts, 'carrier_byteexact_sm86_sm89': json.loads((lab/'iter50-findings.json').read_text())['enabled_images_byteexact'],
       'package_bytes': 0, 'limit_bytes': 8388608, 'ranked_score': None, 'submission': None}
p = lab/'iter50-final-audit.json'
for _ in range(10):
    p.write_text(json.dumps(out, indent=2)+'\n')
    n = sum(p.stat().st_size for p in base.rglob('*') if p.is_file())
    assert n <= out['limit_bytes'], n
    if out['package_bytes'] == n:
        break
    out['package_bytes'] = n
print(json.dumps(out, indent=2))
