from pathlib import Path
import subprocess
import sys
p=Path(__file__).resolve().parent
with (p/'audit.log').open('w') as log:
    subprocess.run([str(p/'audit')],stdout=log,stderr=subprocess.STDOUT,check=True)
print((p/'audit.log').read_text(),flush=True)
for variant,label in [('baseline','-a'),('candidate','-b')]:
    subprocess.run([sys.executable,str(p/'trial.py'),variant,'--label='+label],check=True)
