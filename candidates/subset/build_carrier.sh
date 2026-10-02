#!/usr/bin/env bash
# Regenerate qsb_carrier_sm89.h, the native sm_89 images that QsbCarrier.h loads.
# Adapted from Ryun1's pinning-track build_carrier.sh (public submission 25bd990a, GPL-3).
# Development tool, never run by the ranked harness. Rerun after ANY edit to device code
# in this tree (subset.cu, tests/gpu_epochs/tree.cu or a header they include).
#   NVCC=/path/to/cuda-12.8/bin/nvcc ./build_carrier.sh [QSB_ZEROS_N]
# Build the image with the ranked runner's toolkit, CUDA 12.8: a cubin from a newer
# toolkit may fail to load on the runner's driver, and the carrier then stays off.
# NVCC selects the compiler (default: nvcc on PATH); CUOBJDUMP the dump tool (default:
# cuobjdump next to NVCC, else on PATH; any version reads the cubin). Extra nvcc flags
# (e.g. OpenSSL include paths) can be passed in QSB_CARRIER_NVCC_FLAGS.
#
# Multi-arm probe: one cubin per row of ARMS below, all from this same source with that
# row's extra DEVICE-ONLY -D flags (host code is shared by every arm; QsbCarrier.h refuses an
# arm whose knob string differs outside QSB_PROBE_FREE_KNOBS). Arm 0 is the control (base
# flags) and is the image the single-arm path loads. The runtime time-slices the first QSB_ARMS
# arms (QsbCarrier.h / tree.cu); a one-row table gives the base single-image header. Every arm
# must build with 0 spill bytes in every kernel and resolve the same kernels.
set -euo pipefail
cd "$(dirname "$0")"
Z=${1:-24}
NVCC=${NVCC:-nvcc}
if [ -z "${CUOBJDUMP:-}" ]; then
    CUOBJDUMP=cuobjdump
    d=$(dirname "$(command -v "$NVCC")")
    [ -x "$d/cuobjdump" ] && CUOBJDUMP="$d/cuobjdump"
fi
W=$(mktemp -d)   # logs and the raw cubins stay out of the submission directory
"$NVCC" --version | tail -n 2 | head -n 1 > "$W/nvcc.txt"
if ! grep -q "release 12\.8," "$W/nvcc.txt"; then
    echo "build_carrier: WARNING: $(cat "$W/nvcc.txt") is not CUDA 12.8 (the ranked runner's toolkit);" \
         "the image may not load there. Set NVCC to a 12.8 nvcc." >&2
fi
# Probe v1 arm table (device knobs; see notes/probe-v1.md for why each arm).
ARMS=(
  ""                                         # 0 control (the record's flags)
  "-DQSB_S3_HINT_MODE=2"                     # 1 .L2::64B hint on all four 16-byte loads of a cold record (cadamcat)
  ""                                         # 2 control (A/A)
  "-DQSB_R_CBANK=1"                          # 3 R from the constant bank in the paired front and tail
  ""                                         # 4 control (A/A)
  "-DQSB_R_CBANK_TAILS=1"                    # 5 the same for the tails only
  ""                                         # 6 control (A/A)
)
N=${#ARMS[@]}
for ((a = 0; a < N; a++)); do
  mkdir -p "$W/arm$a"
  printf '%s\n' "${ARMS[$a]}" > "$W/arm$a/flags.txt"
  for ((b = 0; b < a; b++)); do          # same flags as an earlier arm (A/A): same deterministic build
    if [[ "${ARMS[$b]}" == "${ARMS[$a]}" ]]; then
      cp "$W/arm$b/c.cubin" "$W/arm$b/ptxas.log" "$W/arm$b/symbols.txt" "$W/arm$b/sass.txt" "$W/arm$a/"
      continue 2
    fi
  done
  # shellcheck disable=SC2086  # the extra flags and the arm flags are deliberately word-split
  "$NVCC" -O3 -DQSB_ZEROS_N="$Z" -DQSB_CARRIER_BUILD=1 -arch=sm_89 -cubin ${QSB_CARRIER_NVCC_FLAGS:-} \
       ${ARMS[$a]} -Xptxas -v -o "$W/arm$a/c.cubin" subset.cu 2> "$W/arm$a/ptxas.log"
  "$CUOBJDUMP" -symbols "$W/arm$a/c.cubin" > "$W/arm$a/symbols.txt"
  "$CUOBJDUMP" -sass "$W/arm$a/c.cubin" > "$W/arm$a/sass.txt"
done
# Arm 0's build products also at the top of $W, where the single-image tools look.
cp "$W/arm0/c.cubin" "$W/arm0/ptxas.log" "$W/arm0/symbols.txt" "$W/arm0/sass.txt" "$W/"
python3 - "$Z" "$W" "$N" <<'PY'
import base64, glob, hashlib, re, sys
zeros = int(sys.argv[1]); W = sys.argv[2]; N = int(sys.argv[3])
want = [  # order must match enum QsbCarrierKernel in QsbCarrier.h
    ("QK_EG",  r"_Z19kernel_epoch_groups\w+"),
    ("QK_BEI", r"_Z23kernel_build_epochs_inc\w+"),
    ("QK_BFF", r"_Z23kernel_build_first_flat\w+"),
    ("QK_DIG", r"_Z13kernel_digest\w+"),
    ("QK_GT",  r"_Z19kernel_build_gtable\w+"),
    ("QK_HEAL", r"_Z19kernel_gt_heal_scan\w+"),
]
arms = []
for a in range(N):
    D = f"{W}/arm{a}"
    flags = open(D + "/flags.txt").read().strip()
    syms = open(D + "/symbols.txt").read()
    want_a = list(want)
    if re.search(r"_Z18kernel_gt_offset_y\w+", syms):   # QSB_YOFF_S images only (enum entry under #if QSB_YOFF_S)
        want_a.append(("QK_YOFF", r"_Z18kernel_gt_offset_y\w+"))
    names = []
    for kid, pat in want_a:
        hits = sorted(set(re.findall(pat, syms)))
        if len(hits) != 1:
            sys.exit(f"build_carrier: arm {a}: {kid} matched {hits}")
        names.append(hits[0])
    if a and names != arms[0]["names"]:
        sys.exit(f"build_carrier: arm {a} resolves other kernels than arm 0: {names}")
    # Build fingerprint plus every global the host uploads with QSB_TO_SYMBOL (tree.cu and
    # window_schedule_shared.cuh). Edit this list if an upload is added or removed.
    for g in ("qsb_carrier_knobs", "QSB_CONST_SCHEDULE", "QSB_U2R", "QSB_U2R_ISO", "QSB_ISO_INVU",
              "QSB_ISO_XNEG", "QSB_U2R_C", "QSB_PUSH_WORDS", "BINOM_C", "WIN3", "QSB_WINDOW_FIRST",
              "QSB_WINDOW_SECOND", "QSB_WINDOW_CLASS", "QSB_FIRST_CLASS", "QSB_LANE_CLASS",
              "QSB_FIRST_UNIQUE", "QSB_FIRST_COUNT"):
        if not re.search(r"\b%s\b" % g, syms):
            sys.exit(f"build_carrier: arm {a}: global {g} missing from image")
    # The hint must be present in the digest kernel (the only table reader on the search path).
    sass = open(D + "/sass.txt").read()
    fn = re.split(r"\n\s*Function : ", sass)
    dig = [b for b in fn if b.startswith(names[3])]
    if not dig or "LTC64B" not in dig[0]:
        sys.exit(f"build_carrier: arm {a}: digest kernel has no LTC64B load")
    log = open(D + "/ptxas.log").read()
    spills = [int(x) for x in re.findall(r"(\d+) bytes spill (?:stores|loads)", log)]
    if not spills or any(spills):
        sys.exit(f"build_carrier: arm {a} [{flags}]: spill bytes {max(spills or [-1])} (every arm must be 0-spill)")
    m = re.search(r"Compiling entry function '" + re.escape(names[3]) + r"'.*?Used (\d+) registers", log, re.S)
    # The search-path kernels must stay within the default 1 KiB per-thread stack limit, so
    # switching arms never makes the runtime grow the context's local-memory reservation.
    hot_stack = 0
    for n in names[:4]:
        ms = re.search(r"Function properties for " + re.escape(n) + r"\s*\n\s*(\d+) bytes stack frame", log)
        if not ms:
            sys.exit(f"build_carrier: arm {a}: no stack frame line for {n}")
        hot_stack = max(hot_stack, int(ms.group(1)))
    if hot_stack > 1024:
        sys.exit(f"build_carrier: arm {a} [{flags}]: {hot_stack} B stack frame in a search kernel (> 1024)")
    img = open(D + "/c.cubin", "rb").read()
    # Switch-dependent checks, keyed on this arm's own knob string (the qsb_carrier_knobs bytes).
    def knob_on(name):
        return re.search(rb"(^|;|\0)" + name.encode() + rb"=1;", img) is not None
    if knob_on("QSB_QMIX_RT") and not re.search(r"\bQSB_QMIX_MASK_C\b", syms):
        sys.exit(f"build_carrier: arm {a}: QSB_QMIX_RT image without the QSB_QMIX_MASK_C global")
    if knob_on("QSB_GATE_FMA_RT") and not re.search(r"\bQSB_GATE_FMA_C\b", syms):
        sys.exit(f"build_carrier: arm {a}: QSB_GATE_FMA_RT image without the QSB_GATE_FMA_C global")
    if knob_on("QSB_CONST_CALLEE"):
        n_ur = dig[0].count("c[0x3][UR")
        if n_ur != 16:
            sys.exit(f"build_carrier: arm {a}: QSB_CONST_CALLEE image has {n_ur} UR-indexed c[0x3] loads in kernel_digest, want 16")
    arms.append({"flags": flags, "names": names, "img": img, "sha": hashlib.sha256(img).hexdigest(),
                 "n_hint": dig[0].count("LTC64B"), "regs": int(m.group(1)) if m else -1,
                 "stack": hot_stack})
combined = arms[0]["sha"] if N == 1 else hashlib.sha256("".join(x["sha"] for x in arms).encode()).hexdigest()
first = {}                         # identical images (A/A arms) share one base64 copy
for a, x in enumerate(arms):
    first.setdefault(x["sha"], a)
def lines_of(img):
    b64 = base64.b64encode(img).decode()
    return [b64[i:i + 120] for i in range(0, len(b64), 120)]
def label(f):
    return f if f else "base flags"
src = hashlib.sha256()
for p in sorted(glob.glob("*.cu") + glob.glob("*.cuh") + glob.glob("*.h") +
                glob.glob("tests/gpu_epochs/*.cuh") + glob.glob("tests/gpu_epochs/*.h") +
                ["tests/gpu_epochs/tree.cu"]):
    if p == "qsb_carrier_sm89.h":
        continue
    src.update(p.encode() + b"\0" + open(p, "rb").read())
tool = open(W + "/nvcc.txt").read().strip()
A0 = arms[0]
with open("qsb_carrier_sm89.h", "w") as f:
    f.write(f"/* GENERATED by build_carrier.sh -- do not edit. {N} native sm_89 image(s) (probe arms) of subset.cu\n")
    f.write(f" * built with -DQSB_CARRIER_BUILD=1 -DQSB_ZEROS_N={zeros} -arch=sm_89 + each arm's flags ({tool}).\n")
    if N == 1:
        f.write(f" * cubin sha256 {combined}; {len(A0['img'])} bytes; digest kernel LTC64B loads: {A0['n_hint']}.\n")
    else:
        f.write(f" * cubin sha256 {combined}; = sha256 of the concatenated per-arm sha256 hex strings.\n")
    for a, x in enumerate(arms):
        f.write(f" * arm {a}: image sha256 {x['sha']}; {len(x['img'])} bytes; digest regs {x['regs']}, "
                f"0 spill; search-kernel stack {x['stack']} B; LTC64B loads {x['n_hint']}; flags: {label(x['flags'])}\n")
    f.write(f" * source sha256 {src.hexdigest()} (every .cu/.cuh/.h of the build but this file). */\n")
    f.write("#pragma once\n#include <stddef.h>\n")
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
    kids = [k for k, _ in want] + (["QK_YOFF"] if len(A0["names"]) > len(want) else [])
    for kid, n in zip(kids, A0["names"]):
        f.write(f'    "{n}", /* {kid} */\n')
    f.write("};\n")
    f.write(f"static const unsigned qsb_carrier_b64_lines = {len(lines_of(A0['img']))};\n")
    f.write("static const char *const *const qsb_carrier_b64 = qsb_carrier_arm_img0;\n")
for a, x in enumerate(arms):
    print(f"arm {a}: {len(x['img'])} B cubin, sha256 {x['sha'][:16]}, digest regs {x['regs']}, 0 spill, "
          f"stack {x['stack']} B, LTC64B loads {x['n_hint']}, flags: {label(x['flags'])}")
print(f"qsb_carrier_sm89.h: {N} arm(s), combined sha256 {combined[:16]}, {tool}")
PY
grep -A2 "Compiling entry function '_Z13kernel_digest" "$W/ptxas.log" | tail -n 2 || true
echo "build logs: $W (per arm: $W/arm<k>/)"
