import hashlib
import itertools
import json
import random
import struct
from pathlib import Path
import subset_family as study

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "benchmark.json").is_file())
(ROOT / "research").mkdir(exist_ok=True)
prob=json.loads((ROOT/'problems/subset.json').read_text())
import crypto
import problem

for ec in (5,6):
    with (ROOT/f'research/family{ec}.bin').open('wb') as f:
        f.write(struct.pack('<II',ec,3))
        for seed in (0,17,83):
            p=dict(prob)
            if seed:
                r=random.Random(seed)
                for k in ('fixed_prefix','tail_section','tx_suffix'):
                    p[k]=r.randbytes(len(bytes.fromhex(prob[k]))).hex()
                p['dummy_sigs']=[r.randbytes(10).hex() for _ in range(150)]
            rows=[bytes.fromhex(x) for x in p['dummy_sigs']]
            f.write(b''.join(rows)+bytes.fromhex(p['tail_section'])+bytes.fromhex(p['tx_suffix']))
            groups=study.families(p,ec)
            if ec==5:
                picks=study.choose(groups,335)
                patterns=[w for g in picks for w in g]
            else:
                patterns=sorted(w for g in groups.values() if len(g)==5 for w in g)
            rng=random.Random(1818+seed)
            es=[tuple(range(ec)),tuple(range(137-ec,137))]
            es += [tuple(sorted(rng.sample(range(137),ec))) for _ in range(6)]
            for early in es:
                prefix=bytes.fromhex(p['fixed_prefix'])+b''.join(rows[i] for i in range(137) if i not in early)
                cut=len(prefix)//64*64
                st=crypto.sha256_midstate(prefix[:cut])
                rem=prefix[cut:]
                f.write(bytes(early)+struct.pack('<8I',*st)+rem.ljust(64,b'\0'))
                for w in patterns:
                    sk=early+w
                    expected=hashlib.sha256(hashlib.sha256(problem.sub_preimage(p,sk)).digest()).digest()
                    f.write(bytes(sk)+expected)
print('Generated independent hashlib fixtures: three byte seeds, boundary and random early sets.')
