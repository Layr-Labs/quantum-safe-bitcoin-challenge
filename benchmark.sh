#!/usr/bin/env bash
# Run the ranked QSB grinding benchmark and write one score file per bench.
#
#   ./benchmark.sh                # both benches
#   ./benchmark.sh pinning        # just pinning  → score-pinning.json
#   ./benchmark.sh subset         # just subset   → score-subset.json
#
# The two benches are separate tracks (see benchmark.json) and are scored
# independently, so a submission may enter either or both. Ranked defaults come
# from harness/config.json; the QSB_* environment overrides below exist for
# local smoke tests and diagnostics only. A bench that fails verification
# writes no score for that bench.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${root}"

benches=("$@")
[[ ${#benches[@]} -gt 0 ]] || benches=(pinning subset)
for bench in "${benches[@]}"; do
  case "${bench}" in
    pinning|subset) ;;
    *) echo "benchmark.sh: unknown bench '${bench}' (expected pinning or subset)" >&2; exit 2 ;;
  esac
done

# Clear the score for each bench we are about to run, before anything fallible,
# so a failed invocation cannot leave an older result to be picked up as new.
# Scores for benches we are NOT running belong to their own track and stay.
for bench in "${benches[@]}"; do rm -f "${root}/score-${bench}.json"; done

cfg="${root}/harness/config.json"
cfgget() { python3 -c "import json,sys;print(json.load(open('${cfg}')).get('$1'))"; }

zeros="${QSB_ZEROS_N:-$(cfgget leading_zero_bits)}"
mode="${QSB_MODE:-fixed_time}"
seconds="${QSB_SECONDS:-$(cfgget max_seconds)}"
hits="${QSB_HITS:-$(cfgget target_hits)}"
max_rel_var="${QSB_MAX_REL_VAR:-$(cfgget max_relative_variance)}"
grinder="${QSB_GRINDER:-$(cfgget grinder)}"
outdir="${QSB_OUTPUT_DIR:-benchmark-results}"

mkdir -p "${outdir}"
outdir="$(cd "${outdir}" && pwd -P)"

# A ranked run grinds a FRESH problem instance, generated after the submission
# is fixed, so hits precomputed against the committed public instance cannot be
# replayed as a fast run. One seed per invocation, so benches run together share
# an instance, and the result reproduces from the recorded seed. Set
# QSB_PROBLEM_SEED to pin it.
seed="${QSB_PROBLEM_SEED:-}"
[[ -n "${seed}" ]] || seed="$(python3 -c 'import secrets;print(secrets.randbelow(1<<31))')"
# In bridge mode this location must stay inside the repository so the
# read-only sandbox can see the generated problem.
probdir="${outdir}/problem"
mkdir -p "${probdir}"
export QSB_PROBLEM_DIR="${probdir}"

echo "benchmark.sh: benches=${benches[*]} N=${zeros} mode=${mode} grinder=${grinder} seed=${seed}"

# Single-bench is the ranked Yukon path. exec replaces this shell so no later
# line can run after the candidate returns — a hostile grinder must not be
# able to rewrite this script's tail and have bash execute it.
if [[ ${#benches[@]} -eq 1 ]]; then
  bench="${benches[0]}"
  out="${outdir}/run-${bench}.json"
  rm -f "${out}" "${outdir}/score-${bench}.json"
  echo "benchmark.sh: === ${bench} ==="
  exec python3 harness/run_benchmark.py \
        --bench "${bench}" \
        --seed "${seed}" \
        --problem-dir "${QSB_PROBLEM_DIR}" \
        --N "${zeros}" \
        --mode "${mode}" \
        --seconds "${seconds}" \
        --hits "${hits}" \
        --max-rel-var "${max_rel_var}" \
        --grinder "${grinder}" \
        --out "${out}" \
        --score-out "${root}/score-${bench}.json" \
        --score-copy "${outdir}"
fi

failed=0
for bench in "${benches[@]}"; do
  out="${outdir}/run-${bench}.json"
  rm -f "${out}" "${outdir}/score-${bench}.json"
  echo "benchmark.sh: === ${bench} ==="
  if python3 harness/run_benchmark.py \
        --bench "${bench}" \
        --seed "${seed}" \
        --problem-dir "${QSB_PROBLEM_DIR}" \
        --N "${zeros}" \
        --mode "${mode}" \
        --seconds "${seconds}" \
        --hits "${hits}" \
        --max-rel-var "${max_rel_var}" \
        --grinder "${grinder}" \
        --out "${out}" \
        --score-out "${root}/score-${bench}.json" \
        --score-copy "${outdir}"; then
    :
  else
    echo "benchmark.sh: ${bench} failed verification — no score written for ${bench}" >&2
    failed=1
  fi
done

exit "${failed}"
