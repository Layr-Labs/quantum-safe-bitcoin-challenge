from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
for variant,label in [("baseline","-a"),("candidate","-b"),("candidate","-c"),("baseline","-d")]:
    subprocess.run([sys.executable,str(p/"trial.py"),variant,"--label="+label],check=True)
