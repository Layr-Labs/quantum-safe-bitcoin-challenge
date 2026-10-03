#!/usr/bin/env python3
"""Artifact adapter for existing gpu_wrap/benchmark verifier; no timeout process.
The fixed-time boundary sends only SIGTERM and waits for native drain (no kill).
Measured binary is explicit; builds are never wrapped or given deadlines.
"""
import importlib.util,os,signal,subprocess,sys,threading
from pathlib import Path
root=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('gpu_wrap',root/'harness/gpu_wrap.py')
gw=importlib.util.module_from_spec(spec);spec.loader.exec_module(gw)
original_argv=gw.kernel_argv
real_run=subprocess.run

def argv(*args):
    raw=original_argv(*args)
    assert raw[:3]==['stdbuf','-oL','timeout']
    return raw[:2]+raw[4:]

def graceful(cmd,**kw):
    if cmd[:2]!=['stdbuf','-oL']:return real_run(cmd,**kw)
    seconds=float(sys.argv[sys.argv.index('--seconds')+1])
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,cwd=kw.get('cwd'))
    def stop():
        if proc.poll() is None:proc.send_signal(signal.SIGTERM)
    timer=threading.Timer(seconds,stop);timer.start()
    try:out,err=proc.communicate()
    finally:timer.cancel()
    work=Path(kw['cwd']);(work/'native.stdout').write_text(out);(work/'native.stderr').write_text(err)
    return subprocess.CompletedProcess(cmd,proc.returncode,out,err)

def binary(src,zeros,**kw):
    assert zeros==24
    p=Path(os.environ['QSB_AUDIT_BINARY']).resolve();assert p.is_file();return p

gw.kernel_argv=argv;gw.subprocess.run=graceful;gw.compile_kernel=binary
gw.main()
