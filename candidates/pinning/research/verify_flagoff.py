import re, subprocess, tempfile, os

ours = open('candidates/pinning/GPUMath.h', encoding='utf-8').read()
orig = subprocess.run(['git','show','HEAD:candidates/pinning/GPUMath.h'],
                      capture_output=True, text=True).stdout

start = ours.find('#ifndef QSB_SHORT_CARRY')
end   = ours.find('#ifndef QSB_FUSE_SQRADDSUB2')
DEFINES = ours[start:end]

def asm_args(src, fname):
    out = []
    for m in re.finditer(r'__device__ __forceinline__ void ' + fname + r'\s*\(', src):
        i = src.find('asm(', m.start())
        if i < 0: continue
        j = src.find(': "=l"', i)
        if j < 0: j = src.find('\n         :', i)
        out.append(src[i:j])
    return out

def expand(seg, flags_off):
    defs = DEFINES
    if flags_off:
        defs = ('#define QSB_RP_MUL_F8 0\n#define QSB_RP_SQR_F8 0\n'
                '#define QSB_MUL_SFQ 0\n') + defs.replace(
                '#define QSB_RP_MUL_F8 1','#define QSB_RP_MUL_F8 0').replace(
                '#define QSB_RP_SQR_F8 1','#define QSB_RP_SQR_F8 0').replace(
                '#define QSB_MUL_SFQ 1','#define QSB_MUL_SFQ 0')
    with tempfile.NamedTemporaryFile('w', suffix='.c', delete=False) as f:
        f.write(defs + '\nchar *x = ' + seg[4:] + ';\n')
        path = f.name
    try:
        r = subprocess.run(['gcc','-E','-P',path], capture_output=True, text=True, timeout=60)
        return re.sub(r'\s+','',r.stdout)
    finally:
        os.unlink(path)

same = True
for fn in ('_ModMultCore','_ModSqr','_ModSqrAddSub2'):
    new_segs = asm_args(ours, fn)
    old_segs = asm_args(orig, fn)
    print(fn, 'new:', len(new_segs), 'orig:', len(old_segs))
    for k,(nseg,oseg) in enumerate(zip(new_segs, old_segs)):
        off_exp = expand(nseg, True)
        orig_exp = expand(oseg, True)   # orig also carries SAS macros — expand identically
        norm = lambda t: t.replace('\\"','').replace('"','').replace('char*x=','')
        ok = norm(off_exp) == norm(orig_exp)
        exp_txt, orig_txt = off_exp, orig_exp
        same &= ok
        print('  body %d flag-off == original: %s' % (k, ok))
        if not ok:
            A, B = norm(exp_txt), norm(orig_txt)
            print('   lens:', len(A), len(B))
            for i,(a,b) in enumerate(zip(A,B)):
                if a!=b:
                    print('   diff @%d: %r vs %r' % (i, A[max(0,i-40):i+40], B[max(0,i-40):i+40]))
                    break
print('ALL RESTORED' if same else 'FLAG-OFF MISMATCH')
