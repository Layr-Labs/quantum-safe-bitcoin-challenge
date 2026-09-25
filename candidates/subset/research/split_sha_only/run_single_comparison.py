from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
subprocess.run([str(p/"audit_single")],check=True)
for variant,label in [("candidate_single","-i"),("candidate_single","-j"),("candidate","-k")]:
    subprocess.run([sys.executable,str(p/"trial.py"),variant,"--label="+label],check=True)
