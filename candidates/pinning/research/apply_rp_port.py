import sys
bs = chr(92)  # backslash
n_ = bs + 'n'  # literal \n as written inside C string literals
t_ = bs + 't'

p = 'candidates/pinning/GPUMath.h'
s = open(p, encoding='utf-8').read()

edits = [
    # _ModMultCore x2: f8 capture -> QSB_MUL_F8_CAP splice
    ('addc.cc.u64 f3, r3, t;' + n_ + t_ + 'addc.u32 f8, 0, 0;' + n_ + t_ + 'mul.wide',
     'addc.cc.u64 f3, r3, t;' + n_ + '" QSB_MUL_F8_CAP "' + t_ + 'mul.wide', 2),
    # _ModMultCore x2: z8 source -> QSB_MUL_Z8 splice (whole instruction macro)
    (n_ + t_ + 'addc.u32 z8, f8, w7;',
     n_ + '" QSB_MUL_Z8 "', 2),
    # _ModMultCore x2: sfq alias -> QSB_MUL_SF_HEAD splice
    ('sfl, sfh;' + n_ + 'mov.u32 sfq, z8;' + n_ + 'mov.b64 sfz, {z0, sfq};' + n_ + 'mul.wide',
     'sfl, sfh;' + n_ + '" QSB_MUL_SF_HEAD "mul.wide', 2),
    # _ModSqr x2: f8 capture -> QSB_SQR_F8_CAP splice
    ('addc.cc.u64 f3, fr3, t;' + n_ + t_ + 'addc.u32 f8, 0, 0;' + n_ + t_ + 'mul.wide',
     'addc.cc.u64 f3, fr3, t;' + n_ + '" QSB_SQR_F8_CAP "' + t_ + 'mul.wide', 2),
    # _ModSqr x2 + _ModSqrAddSub2 x1: z8 source operand -> QSB_SQR_F8SRC
    ('addc.cc.u32 z8, f8, w7;',
     'addc.cc.u32 z8, " QSB_SQR_F8SRC ", w7;', 3),
    # _ModSqrAddSub2: standalone f8 capture lines (multi-line literal style)
    ('        "' + t_ + 'addc.u32 f8, 0, 0;' + n_ + '"',
     '        QSB_SQR_F8_CAP', 2),
]

for old, new, want in edits:
    cnt = s.count(old)
    if cnt != want:
        print('FAIL: count=%d want=%d  %r' % (cnt, want, old[:60]))
        sys.exit(1)
    s = s.replace(old, new)
    print('ok x%d: %r -> %r' % (cnt, old[:45], new[:45]))

open(p, 'w', encoding='utf-8', newline='').write(s)
print('done')
