#!/usr/bin/env python3
"""Structural audit for QSB_SAS_Z9SUB_ALL on the short-carry square / SAS paths."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
mathh = (ROOT / "GPUMath.h").read_text()
cu = (ROOT / "pinning.cu").read_text()

assert "#define QSB_SAS_Z9SUB_ALL 1" in mathh
assert "#define QSB_HOST_GATE 1" in cu
assert "#define QSB_RP_SQR 1" in cu

# Live short-carry ModSqr must use the splice macros, not the hardcoded g8/z9 pair.
sc = re.search(r"#if QSB_SHORT_CARRY\n__device__.*?_ModSqr.*?\n#else\n__device__", mathh, re.S)
assert sc, "short-carry ModSqr block missing"
body = sc.group(0)
assert "QSB_SAS_G8_TAIL" in body
assert "QSB_SAS_Z9INIT" in body
assert "QSB_SAS_SFQ" in body
assert "QSB_SAS_SFC" in body
assert r"addc.u32 g8, 0, 0;\n\tmov.b64 {z0,z1}, f0" not in body
assert "mad.lo.u32 sfq, z9, 977, z8" not in body

sas = re.search(r"void _ModSqrAddSub2.*?(?=\n__device__|\n// -----)", mathh, re.S)
assert sas, "ModSqrAddSub2 missing"
sbody = sas.group(0)
assert "QSB_SAS_Z8Z9" in sbody
assert "QSB_SAS_SUB_TAIL" in sbody
assert "QSB_SAS_Z9ADD_TAIL" in sbody
assert "QSB_SAS_SFQ" in sbody
# hardcoded mad.lo must not remain in the live SAS fold
assert "mad.lo.u32 sfq, z9, 977, z8" not in sbody

# Flag-off restores the mad.lo form in the macro else branch
assert '#define QSB_SAS_SFQ "mad.lo.u32 sfq, z9, 977, z8;"' in mathh
print("PASS: QSB_SAS_Z9SUB_ALL splices present; live short-carry square/SAS have no hardcoded z9 fold")
