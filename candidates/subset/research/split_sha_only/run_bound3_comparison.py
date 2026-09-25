from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
subprocess.run([str(p/"audit_bound3")],check=True)
for variant,label in [("candidate_bound3","-l"),("candidate_bound3","-m"),("candidate","-n")]:
    subprocess.run([sys.executable,str(p/"trial.py"),variant,"--label="+label],check=True)
