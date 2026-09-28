// CPU differential test of the actual unrolled operation order in the header.
// It does not compile CUDA, measure GPU speed, or replace the official verifier.
import {readFileSync} from 'node:fs';
import {strict as assert} from 'node:assert';
const text = readFileSync(new URL('./sha_schedule_interleaved.cuh', import.meta.url), 'utf8');
const body = text.split('#define QSB_SHA_INTERLEAVED_16(base) do {')[1].split('} while (0)')[0];
const ops = [...body.matchAll(/QSB_SHA_(EXPAND|CONSUME)\(([^)]+)\)/g)]
  .map((m) => ({kind:m[1], args:m[2].split(',')}));
assert.equal(ops.length,32);
let seed=0x937de121;
function next() { seed^=seed<<13; seed^=seed>>>17; seed^=seed<<5; return seed>>>0; }
const rotr=(x,n)=>(x>>>n)|(x<<(32-n));
const s0=x=>rotr(x,7)^rotr(x,18)^(x>>>3);
const s1=x=>rotr(x,17)^rotr(x,19)^(x>>>10);
const S0=x=>rotr(x,2)^rotr(x,13)^rotr(x,22);
const S1=x=>rotr(x,6)^rotr(x,11)^rotr(x,25);
function expand(w,j) { w[j]=(w[j]+s1(w[(j+14)&15])+w[(j+9)&15]+s0(w[(j+1)&15]))>>>0; }
function round(r,names,k,w) {
  const [a,b,c,d,e,f,g,h]=names;
  const t1=(r[h]+S1(r[e])+((r[e]&r[f])^(~r[e]&r[g]))+k+w)>>>0;
  const t2=(S0(r[a])+((r[a]&r[b])^(r[a]&r[c])^(r[b]&r[c])))>>>0;
  r[d]=(r[d]+t1)>>>0; r[h]=(t1+t2)>>>0;
}
const letters='abcdefgh'.split('');
function run(w,r,k,pipeline) {
  for(let segment=0;segment<2;segment++) {
    if(pipeline) {
      for(const op of ops) {
        const j=Number(op.args[0]);
        if(op.kind==='EXPAND') expand(w,j);
        else round(r,op.args.slice(1,9),k[segment*16+j],w[j]);
      }
    } else {
      for(let j=0;j<16;j++) {
        expand(w,j);
        round(r,letters.map((_,i)=>letters[(i-j+16)%8]),k[segment*16+j],w[j]);
      }
    }
  }
  return {w,r};
}
for(let trial=0;trial<20000;trial++) {
  const edge=trial<4 ? [0,0xffffffff,0x80000000,0x55555555][trial] : undefined;
  const w=Array.from({length:16},()=>edge??next());
  const r=Object.fromEntries(letters.map(l=>[l,edge??next()]));
  const k=Array.from({length:32},next);
  assert.deepEqual(run([...w],{...r},k,true),run([...w],{...r},k,false),`trial ${trial}`);
}
console.log('PASS: 20,000 edge/random cases; two consecutive 16-round segments; exact state and schedule equality. GPU compilation and performance not tested.');
