"""Local, unranked A/B trial; all generated artifacts stay in this directory."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "harness"))
from gpu_wrap import compile_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "candidate", "baseline"])
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    if args.action == "build":
        for name in ("candidate", "baseline"):
            compile_kernel(HERE / f"{name}.cu", 24)
        return
    name = args.action
    outdir = HERE / f"output-{name}{args.label}"
    outdir.mkdir(exist_ok=True)
    command = [
        sys.executable, str(ROOT / "harness/run_benchmark.py"),
        "--bench", "subset", "--N", "24", "--mode", "fixed_time",
        "--seconds", "60", "--max-rel-var", "none", "--seed", "2026092501",
        "--out", str(outdir / "run.json"),
        "--problem-dir", str(outdir / "problems"),
        "--grinder", f"cmd:python3 {ROOT / 'candidates/subset/research/stage_profile/capture.py'} --src {HERE / (name + '.cu')} --no-build",
    ]
    with (outdir / "harness.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print((outdir / "harness.log").read_text(), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    artifact = json.loads((outdir / "run.json").read_text())
    print(json.dumps({k: artifact.get(k) for k in (
        "verified_hits", "elapsed_s", "throughput_Mps", "problem_seed", "score"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
