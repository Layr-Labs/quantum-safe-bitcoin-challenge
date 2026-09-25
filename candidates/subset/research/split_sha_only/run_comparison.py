from pathlib import Path
import subprocess
import sys
p=Path(__file__).resolve().parent
for variant,label in [('baseline','-a'),('candidate','-b'),('candidate','-c')]:
    subprocess.run([sys.executable,str(p/'trial.py'),variant,'--label='+label],check=True)
