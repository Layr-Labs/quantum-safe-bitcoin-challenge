from pathlib import Path
import re,random,json
w=Path(__file__).resolve().parent
s=(w.parent/'Serial16Special.cuh').read_text();B=1<<256;N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141;K=65535*(1<<240);mask=(1<<64)-1
rows=[]
for cond,kind in re.findall(r'if \((.*?)\) return ([123])u;',s):
 pairs=re.findall(r'M\[(\d)\]==0x([0-9a-f]+)ULL',cond);assert len(pairs)==4
 rows.append(([int(v,16) for _,v in pairs],int(re.search(r'negative==(\d)u',cond)[1]),int(kind)))
assert len(rows)==3
rng=random.Random(180016);ks=[0,N,K,N-K,1,N-1,B-1]+[rng.randrange(B) for _ in range(100000)]
for k in ks:
 d=2*(k%N)-N;words=[((d%B)>>(64*i))&mask for i in range(4)];sign=int(d<0)
 found=[kind for ws,neg,kind in rows if ws==words and neg==sign]
 expected={0:1,K:2,N-K:3}.get(k%N,0)
 assert found==([expected] if expected else [])
for ws,neg,kind in rows:assert not any(ws==ww and nn==1-neg for ww,nn,kk in rows)
result={'status':'PASS','source_parsed_classifier_cases':len(ks),'wrong_sign_controls':3}
(w/'special-result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
