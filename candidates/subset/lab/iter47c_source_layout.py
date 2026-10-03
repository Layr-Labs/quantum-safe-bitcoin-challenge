#!/usr/bin/env python3
"""Source-only layout transform, preserving strings/newlines and notices.
Original source closure is retained losslessly in the iter47c evidence archive.
CPU host header also uses the existing comment codec after its leading notice.
"""
from pathlib import Path
import re
from iter47b_source_codec import compact

COMMENT_FILES={'tests/gpu_epochs/host_producers_v1.h',
               'tests/gpu_epochs/host_producers_v3.h',
               'tests/gpu_epochs/tree_inverse.cuh',
               'tests/gpu_epochs/pair_shared.cuh'}
def comments(s):
    # Retain the leading notices and any inline attribution/license/site marker.
    keep=s.find('#');keep=keep if keep>=0 else 0
    pat=re.compile(r'copyright|licen[cs]e|gpl|attribution|derived|adapted|'
                   r'dun999|ryun|vanitysearch|BEGIN|END',re.I)
    out=[];i=0
    while i<len(s):
        if s[i] in ('"',"'"):
            q=s[i];j=i+1
            while j<len(s):
                if s[j]=='\\':j+=2;continue
                if s[j]==q:j+=1;break
                j+=1
            out.append(s[i:j]);i=j
        elif s.startswith('//',i) or s.startswith('/*',i):
            j=s.find('\n',i) if s.startswith('//',i) else s.index('*/',i)+2
            j=len(s) if j<0 else j
            part=s[i:j]
            out.append(part if i<keep or pat.search(part) else ' '+'\n'*part.count('\n'))
            i=j
        else:out.append(s[i]);i+=1
    return ''.join(out)

def layout(s, cpu=False, rel=''):
    if cpu:
        k=s.index('*/')+2
        s=s[:k]+compact(s[k:])
    if rel in COMMENT_FILES:s=comments(s)
    out=[];i=0;line_start=True
    while i<len(s):
        if s.startswith('R"',i):
            j=s.index('(',i+2);delimiter=s[i+2:j]
            assert len(delimiter)<=16 and not any(c.isspace() for c in delimiter)
            end=s.index(')'+delimiter+'"',j)+len(delimiter)+2
            part=s[i:end];out.append(part);line_start=part.endswith('\n');i=end
        elif s[i] in ('"',"'"):
            q=s[i];j=i+1
            while j<len(s):
                if s[j]=='\\':j+=2;continue
                if s[j]==q:j+=1;break
                j+=1
            out.append(s[i:j]);line_start=False;i=j
        elif s.startswith('//',i):
            j=s.find('\n',i);j=len(s) if j<0 else j
            out.append(s[i:j]);line_start=False;i=j
        elif s.startswith('/*',i):
            j=s.index('*/',i+2)+2
            part=s[i:j];out.append(part);line_start=False;i=j
        elif s[i] in ' \t\r\f\v':
            j=i+1
            while j<len(s) and s[j] in ' \t\r\f\v':j+=1
            if not line_start and j<len(s) and s[j]!='\n':out.append(' ')
            i=j
        else:
            out.append(s[i]);line_start=s[i]=='\n';i+=1
    result=''.join(out)
    assert result.count('\n')==s.count('\n')
    return result

if __name__=='__main__':
    import argparse,json
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
    base=Path(__file__).resolve().parent.parent;rows={}
    for p in base.rglob('*'):
        if not p.is_file() or p.suffix not in ('.h','.cuh','.cu'):continue
        s=p.read_text();v=layout(s,p.name=='CpuGrindSubset.h',p.relative_to(base).as_posix())
        if v!=s:
            rows[p.relative_to(base).as_posix()]=len(s.encode())-len(v.encode())
            if args.apply:p.write_text(v)
    print(json.dumps({'files':rows,'saved_bytes':sum(rows.values())},indent=2))
