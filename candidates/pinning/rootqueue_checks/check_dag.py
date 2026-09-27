"""Source-linked CUDA event-generation/stream order model; no native compiler."""
from pathlib import Path
import json,re,random,hashlib
D=Path(__file__).resolve().parent;P=D.parent;s=(P/'pinning.cu').read_text();ring=int(re.search(r'^#define QSB_SUBRING (\d+)',s,re.M)[1]);assert ring==6
launch=s[s.index('static void qsb_subpipe_launch('):s.index('/* ============================================================\n * Fixed-base table construction',s.index('static void qsb_subpipe_launch('))]
for x in ['cudaStream_t root_st = P.rtb[P.root_serial ? 0 : (P.g & 1ull)]','cudaStreamWaitEvent(root_st, P.ev_s0[r], 0)','cudaEventRecord(P.ev_rt[r], root_st)','cudaStreamWaitEvent(P.s2, P.ev_rt[r], 0)','cudaStreamWaitEvent(s0, P.ev_s2[r], 0)','cudaEventRecord(P.ev_s2[r], P.s2)','cudaStreamWaitEvent(st, P.ev_out, 0)','cudaStreamWaitEvent(st, P.ev_out2, 0)']:assert x in launch,x
assert not re.search(r'P\.rt\b',launch)
class Dag:
 def __init__(self):self.last={};self.events={};self.parents=[];self.labels=[];self.done=[]
 def put(self,stream,label,extra=()):
  deps=set(extra)
  if stream in self.last:deps.add(self.last[stream])
  n=len(self.parents);assert all(d<n for d in deps);self.parents.append(deps);self.labels.append(label);self.last[stream]=n;return n
 def record(self,stream,event):
  n=self.put(stream,'record '+str(event));self.events[event]=n;return n
 def wait(self,stream,event):
  # CUDA wait snapshots the most recent host-issued event record. Later reuse
  # of the same event cannot redirect an already queued wait.
  assert event in self.events;return self.put(stream,'wait '+str(event),(self.events[event],))
 def ancestor(self,a,b):
  todo=list(self.parents[b]);seen=set()
  while todo:
   x=todo.pop()
   if x==a:return True
   if x not in seen:seen.add(x);todo.extend(self.parents[x])
  return False
 def random_order(self,rng):
  waiting=set(range(len(self.parents)));done=set()
  while waiting:
   choices=[i for i in waiting if self.parents[i]<=done];assert choices,'deadlock';n=rng.choice(choices);done.add(n);waiting.remove(n)
  return len(done)
def case(counts,tail,serial,same_finish,mutate=False,depth=ring):
 q=Dag();g=0;used={};checks=[];all_ranges=[];hostreads=[];rng=random.Random(3104+sum(counts)*31+serial)
 for batch,m in enumerate(counts):
  host=('slot',batch%4);reset=q.put(host,('reset',batch));q.record(host,'input')
  finishstreams=[('f',0),('f',0 if same_finish else 1)]
  q.wait(finishstreams[0],'input')
  if finishstreams[1]!=finishstreams[0]:q.wait(finishstreams[1],'input')
  finals=[];ranges=[]
  for off in range(0,(m-1)*131072+tail,131072):
   n=min(131072,(m-1)*131072+tail-off);r=g%depth;lane=g&1;prepstream=('p',lane);rootstream=('r',0 if serial else lane);finishstream=finishstreams[lane]
   previous=used.get(r)
   if previous is not None and not mutate:q.wait(prepstream,('finish',r))
   prep=q.put(prepstream,('prepare',batch,g,r));q.record(prepstream,('prepare',r));q.wait(rootstream,('prepare',r));root=q.put(rootstream,('root',batch,g,r));q.record(rootstream,('root',r));q.wait(finishstream,('root',r));finish=q.put(finishstream,('finish',batch,g,r));q.record(finishstream,('finish',r));used[r]=finish
   assert q.ancestor(prep,root) and q.ancestor(root,finish) and q.ancestor(reset,finish)
   if previous is not None:assert q.ancestor(previous,prep),'unsafe ring reuse'
   finals.append(finish);ranges.append((off,n,(n+127)//128));g+=1
  q.record(finishstreams[0],'out0');q.wait(host,'out0')
  if finishstreams[1]!=finishstreams[0]:q.record(finishstreams[1],'out1');q.wait(host,'out1')
  read=q.put(host,('readback',batch));assert all(q.ancestor(f,read) for f in finals)
  assert sum(n for off,n,nb in ranges)==(m-1)*131072+tail
  assert [off for off,n,nb in ranges]==list(range(0,(m-1)*131072+tail,131072));assert all(1<=nb<=1024 for off,n,nb in ranges)
  hostreads.append(read);all_ranges.extend(ranges)
 # Exercise alternate legal execution orders in small graphs; large graphs use
 # topologically increasing IDs as a guaranteed schedule plus ancestor checks.
 if len(q.parents)<300:q.random_order(rng)
 return len(q.parents),len(all_ranges)
cases=nodes=batches=0
for serial in [False,True]:
 for same in [False,True]:
  for m in [1,2,3,4,5,6,7,9,17,32]:
   for tail in [1,127,128,129,131071,131072]:
    ns,bs=case([m,m+1,max(1,m-1)],tail,serial,same);nodes+=ns;batches+=bs;cases+=1
negative=0
for serial in [False,True]:
 for same in [False,True]:
  try:case([9,7,8],129,serial,same,mutate=True)
  except AssertionError:negative+=1
assert negative==4
# Dependency-only latency witness: p0 is delayed, p1 has independent completed
# work. No resource model, throughput or GPU speed is inferred from this.
def witness(serial):
 prep=[100,1];rootends=[];last=[0,0]
 for lane in [0,1]:
  qlane=0 if serial else lane;end=max(prep[lane],last[qlane])+2;last[qlane]=end;rootends.append(end)
 return rootends
assert witness(False)==[102,3] and witness(True)==[102,104]
state=131072*4*16;root=1024*8*8;superroot=4*4*8
# checkpoint stride comes from the promoted macro, not an assumed row count.
stride=int(re.search(r'#define QSB_CHECKPOINT_STRIDE\s+(\d+)',s)[1]);ckpt=4*4*stride*8
result=dict(status='PASS_SOURCE_LINKED_EVENT_DAG',cases=cases,microbatches=batches,graph_nodes=nodes,negative_omitted_reuse_caught=negative,tail_sizes=[1,127,128,129,131071,131072],root_queues=2,ring=ring,per_entry_bytes=state+root+superroot+ckpt,added_buffer_bytes=2*(state+root+superroot+ckpt),head_of_line_dependency_witness={'serial_root_completions':witness(True),'parallel_root_completions':witness(False)},native_compile=False,GPU_run=False,throughput_prediction=False)
(D/'dag-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
