from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'harness'))
import gpu_wrap
original=gpu_wrap.subprocess.run
def run(*args,**kwargs):
    result=original(*args,**kwargs)
    if kwargs.get('capture_output'):
        (Path(kwargs['cwd'])/'kernel.log').write_text(result.stdout+result.stderr)
    return result
gpu_wrap.subprocess.run=run
gpu_wrap.main()
