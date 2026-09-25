from pathlib import Path
import subprocess, sys
p=Path(__file__).resolve().parent
for variant,label in [("frontier","-d"),("candidate","-e")]:
    subprocess.run([sys.executable,str(p/"trial.py"),variant,"--label="+label],check=True)
