#!/usr/bin/env bash
# Regenerate qsb_carrier_sm89.h, the native sm_89 images that QsbCarrier.h loads.
# Development tool, never run by the ranked harness. Rerun after ANY edit to
# pinning.cu or a header it includes, with the same CUDA toolkit as the runner (12.8).
#   ./build_carrier.sh [QSB_ZEROS_N]
#
# Multi-arm probe (challenges/qsb-tools/PROBE-DESIGN.md): one cubin per row of ARMS below,
# all from this same source with that row's extra nvcc/ptxas flags. Arm 0 is the control
# (base flags) and is the image the single-arm path loads. The runtime time-slices the arms
# (QsbCarrier.h); a one-row table generates a header whose runtime is the base single-image
# path. Every arm must build with 0 spill bytes in every kernel and resolve the same kernels.
set -euo pipefail
cd "$(dirname "$0")"
Z=${1:-24}
W=$(mktemp -d)   # logs and the raw cubins stay out of the submission directory
# Probe arm table (cefika probe 1, on patternrecognition9-del's multi-arm carrier). Every arm is an
# existing #ifndef-guarded switch of the promoted tree at a value it has not been measured at, except
# arm 5, which adds work to price SHA-256d in prepare (the CPU-z question).
ARMS=(
  ""                                                            # 0 control (base flags)
  "-DQSB_PO_ALU=0"                                              # 1 record part po-alu back off
  "-DQSB_FIN_IVFOLD=1"                                          # 2 finish: IV literal rides in the K+W add (rounds 2-3)
  "-DQSB_FIN_RASSOC=2"                                          # 3 finish: T2 formed beside T1 (reassociated adds)
  "-DQSB_CHAIN_ALU=1"                                           # 4 chain-end carries via constant-bank zero (IADD3.X)
  "-DQSB_PROBE_SHA2X=1"                                         # 5 two extra SHA-256 compressions per candidate (cost probe)
  ""                                                            # 6 control again (A/A)
)
N=${#ARMS[@]}
for ((a = 0; a < N; a++)); do
  mkdir -p "$W/arm$a"
  # shellcheck disable=SC2086  # the arm flags are deliberately word-split
  nvcc -O3 -DQSB_ZEROS_N="$Z" -DQSB_CARRIER_BUILD=1 -arch=sm_89 -cubin \
       -Xptxas -v ${ARMS[$a]} -o "$W/arm$a/c.cubin" pinning.cu 2> "$W/arm$a/ptxas.log"
  cuobjdump -symbols "$W/arm$a/c.cubin" > "$W/arm$a/symbols.txt"
  cuobjdump -sass "$W/arm$a/c.cubin" > "$W/arm$a/sass.txt"
  printf '%s\n' "${ARMS[$a]}" > "$W/arm$a/flags.txt"
done
# Arm 0's build products also at the top of $W, where the single-image tools look.
cp "$W/arm0/c.cubin" "$W/arm0/ptxas.log" "$W/arm0/symbols.txt" "$W/arm0/sass.txt" "$W/"
python3 - "$Z" "$W" "$N" <<'PY'
import base64, hashlib, re, sys
zeros = int(sys.argv[1]); W = sys.argv[2]; N = int(sys.argv[3])
want = [  # order must match enum QsbCarrierKernel in QsbCarrier.h
    ("QK_S0",    r"_Z23kernel_pinning_pipelineILb1ELi0EE\w+"),
    ("QK_S2",    r"_Z23kernel_pinning_pipelineILb1ELi2EE\w+"),
    ("QK_RGP",   r"_Z22qsb_root_group_prepare\w+"),
    ("QK_ISR",   r"_Z22qsb_invert_super_roots\w+"),
    ("QK_RGF",   r"_Z21qsb_root_group_finish\w+"),
    ("QK_BUILD", r"_Z19kernel_build_gtable\w+"),
    ("QK_YOFF",  r"_Z18qsb_table_offset_y\w+"),
    ("QK_RF",    r"_Z14qsb_root_fusedILi\d+EE\w+"),        # optional: absent when QSB_ROOT_FUSED=0
    ("QK_RR",    r"_Z17qsb_root_register\w+"),             # optional
    ("QK_PFC",   r"_Z29qsb_prefix_field_check_kernel\w+"),  # optional
]
optional = {"QK_RF", "QK_RR", "QK_PFC"}
arms = []
for a in range(N):
    D = f"{W}/arm{a}"
    flags = open(D + "/flags.txt").read().strip()
    syms = open(D + "/symbols.txt").read()
    names = []
    for kid, pat in want:
        hits = sorted(set(re.findall(pat, syms)))
        if len(hits) == 0 and kid in optional:
            names.append("")
            continue
        if len(hits) != 1:
            sys.exit(f"build_carrier: arm {a}: {kid} matched {hits}")
        names.append(hits[0])
    for g in ("qsb_carrier_zeros", "pin_u2rx_words", "pin_u2ry_words", "pin_iso_invu_words",
              "pin_iso_u2ry_words", "pin_iso_xneg", "pin_recovery_c", "pin_u2rk_words",
              "pin_one_mul", "pin_zero_add", "pin_tail_words"):
        if not re.search(r"\b%s\b" % g, syms):
            sys.exit(f"build_carrier: arm {a}: global {g} missing from image")
    if a and names != arms[0]["names"]:
        sys.exit(f"build_carrier: arm {a} resolves other kernels than arm 0: {names}")
    # The hint must be present in the prepare kernel (the only table reader on the hot path).
    sass = open(D + "/sass.txt").read()
    fn = re.split(r"\n\s*Function : ", sass)
    s0 = [b for b in fn if b.startswith(names[0])]
    if not s0 or "LTC64B" not in s0[0]:
        sys.exit(f"build_carrier: arm {a}: prepare kernel has no LDG.E.LTC64B load")
    log = open(D + "/ptxas.log").read()
    spills = [int(x) for x in re.findall(r"(\d+) bytes spill (?:stores|loads)", log)]
    if not spills or any(spills):
        sys.exit(f"build_carrier: arm {a} [{flags}]: spill bytes {max(spills or [-1])} (every arm must be 0-spill)")
    m = re.search(r"Compiling entry function '" + re.escape(names[0]) + r"'.*?Used (\d+) registers", log, re.S)
    # Kernels launched per slice (S0, S2 and the three root kernels) must stay within the
    # default 1 KiB per-thread stack limit, so switching arms never makes the runtime grow the
    # context's local-memory reservation after the batch was sized to the free VRAM.
    hot_stack = 0
    for n in names[:5]:
        ms = re.search(r"Function properties for " + re.escape(n) + r"\s*\n\s*(\d+) bytes stack frame", log)
        if not ms:
            sys.exit(f"build_carrier: arm {a}: no stack frame line for {n}")
        hot_stack = max(hot_stack, int(ms.group(1)))
    if hot_stack > 1024:
        sys.exit(f"build_carrier: arm {a} [{flags}]: {hot_stack} B stack frame in a search kernel (> 1024)")
    img = open(D + "/c.cubin", "rb").read()
    arms.append({"flags": flags, "names": names, "img": img, "sha": hashlib.sha256(img).hexdigest(),
                 "n_hint": s0[0].count("LTC64B"), "regs": int(m.group(1)) if m else -1,
                 "stack": hot_stack})
combined = hashlib.sha256("".join(x["sha"] for x in arms).encode()).hexdigest()
first = {}                         # identical images (A/A arms) share one base64 copy
for a, x in enumerate(arms):
    first.setdefault(x["sha"], a)
def lines_of(img):
    b64 = base64.b64encode(img).decode()
    return [b64[i:i + 120] for i in range(0, len(b64), 120)]
def label(f):
    return f if f else "base flags"
A0 = arms[0]
with open("qsb_carrier_sm89.h", "w") as f:
    f.write(f"/* GENERATED by build_carrier.sh -- do not edit. {N} native sm_89 images (probe arms) of pinning.cu\n")
    f.write(f" * built with -DQSB_CARRIER_BUILD=1 -DQSB_ZEROS_N={zeros} -arch=sm_89 + each arm's flags.\n")
    f.write(f" * cubin sha256 {combined}; = sha256 of the concatenated per-arm sha256 hex strings.\n")
    for a, x in enumerate(arms):
        f.write(f" * arm {a}: image sha256 {x['sha']}; {len(x['img'])} bytes; prepare regs {x['regs']}, "
                f"0 spill; search-kernel stack {x['stack']} B; LTC64B loads {x['n_hint']}; flags: {label(x['flags'])}\n")
    f.write(" */\n#pragma once\n#include <stddef.h>\n")
    f.write(f"#define QSB_CARRIER_ARMS {N}\n")
    for a, x in enumerate(arms):
        if first[x["sha"]] != a:
            continue
        f.write(f"static const char *const qsb_carrier_arm_img{a}[] = {{\n")
        for l in lines_of(x["img"]):
            f.write(f'"{l}",\n')
        f.write("};\n")
    def arr(ctype, name, vals):
        f.write(f"static const {ctype} {name}[{N}] = {{" + ", ".join(vals) + "};\n")
    arr("size_t", "qsb_carrier_arm_bytes", [str(len(x["img"])) for x in arms])
    arr("unsigned", "qsb_carrier_arm_b64_lines", [str(len(lines_of(x["img"]))) for x in arms])
    arr("char *const", "qsb_carrier_arm_sha256", [f'"{x["sha"]}"' for x in arms])
    arr("char *const", "qsb_carrier_arm_flags", [f'"{label(x["flags"])}"' for x in arms])
    arr("char *const *const", "qsb_carrier_arm_b64", [f"qsb_carrier_arm_img{first[x['sha']]}" for x in arms])
    f.write("/* Arm 0 (control) under the single-image names of the base loader. */\n")
    f.write(f"static const size_t qsb_carrier_cubin_bytes = {len(A0['img'])};\n")
    f.write(f'static const char qsb_carrier_cubin_sha256[] = "{A0["sha"]}";\n')
    f.write("static const char *const qsb_carrier_kernel_names[] = {\n")
    for kid, n in zip([k for k, _ in want], A0["names"]):
        f.write(f'    "{n}", /* {kid} */\n')
    f.write("};\n")
    f.write(f"static const unsigned qsb_carrier_b64_lines = {len(lines_of(A0['img']))};\n")
    f.write("static const char *const *const qsb_carrier_b64 = qsb_carrier_arm_img0;\n")
for a, x in enumerate(arms):
    print(f"arm {a}: {len(x['img'])} B cubin, sha256 {x['sha'][:16]}, prepare regs {x['regs']}, 0 spill, "
          f"LTC64B loads {x['n_hint']}, flags: {label(x['flags'])}")
print(f"qsb_carrier_sm89.h: {N} arms, combined sha256 {combined[:16]}")
PY
grep -E "Function properties for _Z23kernel_pinning_pipelineILb1ELi[02]|registers" "$W/ptxas.log" | grep -A1 "kernel_pinning_pipelineILb1" | head -8 || true
echo "build logs: $W (per arm: $W/arm<k>/)"
