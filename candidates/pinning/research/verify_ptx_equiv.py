import re, subprocess, tempfile, os

def get_fn_text(src, name, n):
    """Return the full text of the nth 'void name(' function incl. defines it needs."""
    starts = [m.start() for m in re.finditer(re.escape(name), src)]
    return starts

ours = open('candidates/pinning/GPUMath.h', encoding='utf-8').read()
fkn  = open('.fk_new_gm.h', encoding='utf-8', errors='replace').read()

# Extract the shared defines block from OUR file (SHORT_CARRY + SAS + RP + SFQ)
start = ours.find('#ifndef QSB_SHORT_CARRY')
end   = ours.find('#ifndef QSB_FUSE_SQRADDSUB2')
DEFINES = ours[start:end]
assert 'QSB_SHORT_CARRY' in DEFINES and 'QSB_MUL_Z8' in DEFINES, 'defines slice incomplete'

# For each function body, pull the asm(...) argument with macro splices,
# run it through the preprocessor with the defines, compare expansion.
def asm_args(src, fname):
    out = []
    for m in re.finditer(r'__device__ __forceinline__ void ' + fname + r'\s*\(', src):
        i = src.find('asm(', m.start())
        if i < 0: continue
        # find the closing paren of asm( ... ) — scan for '\n' + ':' constraints
        j = src.find(': "=l"', i)
        if j < 0: j = src.find('\n         :', i)
        seg = src[i:j]
        out.append(seg)
    return out

def expand(seg):
    with tempfile.NamedTemporaryFile('w', suffix='.c', delete=False) as f:
        f.write(DEFINES + '\nchar *x = ' + seg[4:] + ';\n')
        path = f.name
    try:
        r = subprocess.run(['gcc', '-E', '-P', path], capture_output=True, text=True, timeout=60)
        return re.sub(r'\s+', '', r.stdout)
    finally:
        os.unlink(path)

same = True
for fn in ('_ModMultCore', '_ModSqr', '_ModSqrAddSub2'):
    o_segs = asm_args(ours, fn)
    f_segs = asm_args(fkn, fn)
    print(fn, 'ours:', len(o_segs), 'fk:', len(f_segs))
    for k,(a,b) in enumerate(zip(o_segs, f_segs)):
        ea, eb = expand(a), expand(b)
        ok = ea == eb
        same &= ok
        print('  body %d expanded-identical: %s' % (k, ok))
        if not ok:
            for i,(ca,cb) in enumerate(zip(ea,eb)):
                if ca!=cb:
                    print('   first diff @%d: %r vs %r' % (i, ea[max(0,i-40):i+40], eb[max(0,i-40):i+40]))
                    break
print('ALL IDENTICAL' if same else 'DIFFERENCES FOUND')
