"""Run short unscored diagnostic profiles under the caller's GPU lock."""
from pathlib import Path
import subprocess

P = Path(__file__).resolve().parent
for label in ("paired", "bound3", "single"):
    out = P / f"profile-output-{label}"
    out.mkdir(exist_ok=True)
    with (out / "profile.log").open("w") as log:
        subprocess.run([
            "timeout", "90", "stdbuf", "-oL", str(P / f"profile_{label}"),
            str(P / "profile-problem.bin"), "0", "279549145", "2399405719",
            "1", "0", "single_hash",
        ], cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(label + " profile complete", flush=True)
