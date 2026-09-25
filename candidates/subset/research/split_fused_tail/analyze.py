"""Compare verified runs over the same completed lexicographic epoch prefix."""
from pathlib import Path
import gzip
import json
import math
import re
import sys


def read_run(folder):
    path=folder/'run.json'
    if path.exists():return json.loads(path.read_text())
    return json.loads(gzip.decompress((folder/'run.json.gz').read_bytes()))


def completed(folder):
    text=(folder/'results/digest_summary_gpu0.txt').read_text()
    counts=[int(v) for v in re.findall(r'total_attempts=(\d+)',text)]
    assert counts
    return max(counts)


def epoch_rank(omissions):
    c=omissions[:6]
    assert len(c)==6 and c==sorted(c) and len(set(c))==6 and c[-1]<137
    rank=0
    previous=-1
    for i,value in enumerate(c):
        rank+=math.comb(137-previous-1,6-i)-math.comb(137-value,6-i)
        previous=value
    return rank


def compare(baseline,candidate):
    b,c=read_run(baseline),read_run(candidate)
    assert b['score']['valid'] and c['score']['valid']
    assert b['problem_seed']==c['problem_seed'] and b['zeros_n']==c['zeros_n']
    bc,cc=completed(baseline),completed(candidate)
    assert bc%128==0 and cc%128==0
    epochs=min(bc,cc)//128
    def prefix(v):
        return {(tuple(h['skip']),h['recid']) for h in v['hits'] if epoch_rank(h['skip'])<epochs}
    bh,ch=prefix(b),prefix(c)
    return {'baseline_verified_hits':b['verified_hits'],'candidate_verified_hits':c['verified_hits'],
            'baseline_verified_Mps':b['throughput_Mps'],'candidate_verified_Mps':c['throughput_Mps'],
            'verified_gain_percent':100*(c['throughput_Mps']/b['throughput_Mps']-1),
            'baseline_completed':bc,'candidate_completed':cc,
            'completed_gain_percent':100*(cc/bc-1),'common_epochs':epochs,
            'baseline_common_hits':len(bh),'candidate_common_hits':len(ch),
            'common_prefix_exact':bh==ch,'baseline_only_hits':len(bh-ch),'candidate_only_hits':len(ch-bh),
            'hardware':'local RTX 3090, not official RTX 4090'}


if __name__=='__main__':
    print(json.dumps(compare(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
