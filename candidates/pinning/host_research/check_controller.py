from pathlib import Path
import json
s=(Path(__file__).resolve().parent.parent/'cpu_cogrind.h').read_text()
assert 'const int on = C.ab_left == 4 || C.ab_left == 1;' in s
assert 'set_allowed(C.ab_left == 1 ? C.wmax : 0)' in s
assert 'now - C.win_t0 < 2.0' in s
assert 'C.ab_cpu_done += cd - C.win_cand0;' in s
assert 'const int bad = loss > cpu_frac + 0.003;' in s
assert 'loss > 0.015' not in s
states=[4,3,2,1];pattern=[x==4 or x==1 for x in states];assert pattern==[True,False,False,True]
# Synthetic exact-time model: equal-weight windows cancel linear background
# drift. No claim about thermal hysteresis, nonlinear drift or hardware noise.
checks=0
for drift in [-.01,-.005,0,.005,.01]:
 for loss in [0,.004,.008,.012,.02,.05]:
  for cpu_fraction in [0,.002,.005,.015,.04]:
   background=[1+drift*t for t in [-3,-1,1,3]]
   intervals=[t*(1+loss if on else 1) for t,on in zip(background,pattern)]
   on_t=(intervals[0]+intervals[3])/2;off_t=(intervals[1]+intervals[2])/2
   got_loss=on_t/off_t-1
   assert abs(got_loss-loss)<1e-12
   cpu_rate=cpu_fraction/on_t
   measured_cpu_fraction=cpu_rate*on_t
   assert abs(measured_cpu_fraction-cpu_fraction)<1e-12
   if abs(loss-cpu_fraction-.003)>1e-12:assert (got_loss>measured_cpu_fraction+.003)==(loss>cpu_fraction+.003)
   # Net throughput accounting in the same units as GPU candidates.
   net_ratio=(1+cpu_fraction)/(1+loss)
   assert (net_ratio>=1)==(cpu_fraction>=loss)
   checks+=1
# CPU off intervals contribute no count/time to the inferred on-window rate.
rate=1234567;durations=[2.01,2.02,2.03,2.04]
cpu_done=sum(rate*t for on,t in zip(pattern,durations) if on);cpu_time=sum(t for on,t in zip(pattern,durations) if on)
assert abs(cpu_done/cpu_time-rate)<1e-9
# One bad trial does not shed; a good trial resets strikes; counts never negative.
for n in range(1,257):
 for decisions in [[True],[True,False,True],[True,True],[False,True,True],[True]*20]:
  workers=n;strikes=0
  for bad in decisions:
   before=workers
   if bad and strikes>=1:workers=max(0,workers-(workers+3)//4);strikes=0
   elif bad:strikes=1
   else:strikes=0
   assert 0<=workers<=before
  if decisions==[True] or decisions==[True,False,True]:assert workers==n
  if decisions==[True,True]:assert workers==n-(n+3)//4
res={'synthetic_drift_and_contribution_cases':checks,'worker_counts_checked':256,'abba_pattern':pattern,'window_seconds':2,'margin_percentage_points':0.3,'consecutive_bad_trials_required':2,'runtime_test':False}
print(json.dumps(res,indent=2))
