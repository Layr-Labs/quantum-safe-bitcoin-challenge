from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
for variant,label in [("candidate128","-f"),("candidate128","-g"),("candidate","-h")]:
    subprocess.run([sys.executable,str(p/"trial.py"),variant,"--label="+label],check=True)
