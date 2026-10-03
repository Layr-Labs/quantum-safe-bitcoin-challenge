#!/usr/bin/env python3
"""Comment-only compaction retaining newlines and the leading attribution block.
Complete original source remains in the measured archive. No executable edit.
"""
def compact(s):
    out=[];i=0
    # Preserve leading contiguous comment attribution up to first preprocessor.
    keep=s.find('#');keep=keep if keep>=0 else 0
    while i<len(s):
        if s[i] in ('"',"'"):
            q=s[i];j=i+1
            while j<len(s):
                if s[j]=='\\':j+=2;continue
                if s[j]==q:j+=1;break
                j+=1
            out.append(s[i:j]);i=j
        elif s.startswith('//',i):
            j=s.find('\n',i);j=len(s) if j<0 else j
            out.append(s[i:j] if i<keep else ' ');i=j
        elif s.startswith('/*',i):
            j=s.find('*/',i+2);assert j>=0;j+=2
            out.append(s[i:j] if i<keep else ' '+('\n'*s[i:j].count('\n')));i=j
        else:out.append(s[i]);i+=1
    return '\n'.join(line.rstrip() for line in ''.join(out).split('\n'))
if __name__=='__main__':
    from pathlib import Path
    p=Path('candidates/subset/tests/gpu_epochs/tree.cu');s=p.read_text();v=compact(s)
    assert s.count('\n')==v.count('\n');p.write_text(v);print('Comment bytes retired',len(s)-len(v))
