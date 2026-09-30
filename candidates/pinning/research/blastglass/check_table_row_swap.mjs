// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 BlastGlass
// Original, independent CPU model of build_segment's buffer ownership.
// Synthetic arithmetic is intentional: this does not execute the candidate,
// CUDA, secp256k1, the benchmark verifier, or a C++ compilation.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
const sourcePath=fileURLToPath(new URL('../../cg_table.h', import.meta.url));
const source=fs.readFileSync(sourcePath,'utf8');
const highfold=source.split('#if QSB_CG_HIGHFOLD')[1].split('#else')[0];
const rowSize=Number(highfold.match(/#define TB_R (\d+)/)[1]);
const segmentSize=Number(highfold.match(/#define QSB_CG_SEG (\d+)/)[1]);
assert.equal(rowSize,4096);assert.equal(segmentSize,65536);
assert.match(source,/fe4_t \*oldx = rowx; rowx = nx; nx = oldx;/);
assert.match(source,/fe4_t \*oldy = rowy; rowy = ny; ny = oldy;/);
const M=0xffffffffn;
const syntheticAdd=(px,py,qx,qy)=>[(px*3n+py*5n+qx*7n+qy*11n)&M,(px*13n+py*17n+qx*19n+qy*23n)&M];
function worker(R){
  const regions=Array.from({length:9},()=>new Array(R).fill(-1n));
  assert.equal(new Set(regions).size,9);
  return {regions,table:[],copyBytes:0,swaps:0};
}
function buildModel(W,{R,ne,j,seg,errorAt=-1},useSwap){
  const [rx0,ry0,nx0,ny0,kx,ky,tmpNum,tmpDen,tmpPrefix]=W.regions;
  let rx=rx0,ry=ry0,nx=nx0,ny=ny0;
  const args=[rx,ry,nx,ny,kx,ky,tmpNum,tmpDen,tmpPrefix];
  for(let i=0;i<R;i++){kx[i]=BigInt(1000*j+i+1);ky[i]=BigInt(2000*j+3*i+1);}
  const constants=[kx.slice(),ky.slice()];
  const firstCount=Math.min(ne,R);
  const sx=BigInt(17*j+29*seg+1),sy=BigInt(31*j+43*seg+2);
  let status=0;
  if(j>=1&&seg===0){
    for(let i=0;i<firstCount;i++){rx[i]=kx[i];ry[i]=ky[i];}
  }else{
    rx[0]=sx;ry[0]=sy;
    for(let i=1;i<firstCount;i++){nx[i-1]=sx;ny[i-1]=sy;}
    if(firstCount>1&&errorAt==='init')status=-1;
    else for(let i=1;i<firstCount;i++)[rx[i],ry[i]]=syntheticAdd(nx[i-1],ny[i-1],kx[i-1],ky[i-1]);
  }
  let done=0,transitionIndex=0;
  while(status===0){
    const take=Math.min(R,ne-done);
    for(let i=0;i<take;i++)W.table.push([rx[i],ry[i]]);
    done+=take;
    if(done>=ne)break;
    // Coordinate, constant, and three scratch regions remain pairwise disjoint.
    assert.equal(new Set([rx,ry,nx,ny,kx,ky,tmpNum,tmpDen,tmpPrefix]).size,9);
    const input=[rx.slice(),ry.slice()];
    for(let i=0;i<R;i++){tmpNum[i]=(ry[i]-ky[R-1])&M;tmpDen[i]=(rx[i]-kx[R-1])&M;tmpPrefix[i]=BigInt(i);}
    if(transitionIndex===errorAt){status=-1;break;}
    // A successful next-row operation writes every output element. The inputs
    // are unchanged and no input region aliases an output or scratch region.
    for(let i=R-1;i>=0;i--)[nx[i],ny[i]]=syntheticAdd(rx[i],ry[i],kx[R-1],ky[R-1]);
    assert.deepEqual(rx,input[0]);assert.deepEqual(ry,input[1]);
    if(useSwap){[rx,nx]=[nx,rx];[ry,ny]=[ny,ry];W.swaps++;}
    else{for(let i=0;i<R;i++){rx[i]=nx[i];ry[i]=ny[i];}W.copyBytes+=R*64;}
    transitionIndex++;
  }
  assert.deepEqual(kx,constants[0]);assert.deepEqual(ky,constants[1]);
  // Parameters are passed by value: swaps cannot change the caller's original
  // allocation identities, used again when the next segment is initialized.
  for(let i=0;i<9;i++)assert.equal(W.regions[i],args[i]);
  return {status,done};
}
let configurations=0,segmentCalls=0,emittedPoints=0;
for(const R of [1,2,4,32,256,rowSize]){
  const lengths=new Set([1,Math.max(1,R-1),R,R+1,2*R-1,2*R,3*R+1,16*R-1,16*R]);
  for(const ne of lengths){
    for(const errorAt of [-1,'init',0,1,14]){
      const copy=worker(R),swap=worker(R);
      for(const [j,seg,n] of [[0,0,ne],[1,0,1],[1,1,R+1],[2,3,ne],[1,0,R]]){
        const cfg={R,ne:n,j,seg,errorAt};
        assert.deepEqual(buildModel(copy,cfg,false),buildModel(swap,cfg,true));
        assert.deepEqual(copy.table,swap.table);segmentCalls++;
      }
      assert.equal(swap.copyBytes,0);
      emittedPoints+=swap.table.length;configurations++;
    }
  }
}
// Exact copy volume for one successful full segment at the current constants.
const copy=worker(rowSize),swap=worker(rowSize);
const full={R:rowSize,ne:segmentSize,j:1,seg:1};
assert.deepEqual(buildModel(copy,full,false),buildModel(swap,full,true));
assert.deepEqual(copy.table,swap.table);
const expectedCopyBytes=(Math.ceil(segmentSize/rowSize)-1)*rowSize*64;
assert.equal(copy.copyBytes,expectedCopyBytes);assert.equal(swap.copyBytes,0);
// A separate, original toy-curve model exercises doubling and inverse-point
// errors. Curve p=223, y^2=x^3+7 is intentionally not the challenge's curve.
const p=223n,mod=x=>((x%p)+p)%p;
function inv(a){let [r,nr,t,nt]=[p,mod(a),0n,1n];while(nr){const q=r/nr;[r,nr]=[nr,r-q*nr];[t,nt]=[nt,t-q*nt];}return r===1n?mod(t):null;}
function toyAdd(a,b,stats){
  let numerator,denominator;
  if(a[0]===b[0]){
    if(a[1]!==b[1]||a[1]===0n){stats.inverseErrors++;return null;}
    stats.doublings++;numerator=3n*a[0]*a[0];denominator=2n*a[1];
  }else{numerator=b[1]-a[1];denominator=b[0]-a[0];}
  const reciprocal=inv(denominator);if(reciprocal===null)return null;
  const slope=mod(numerator*reciprocal),x=mod(slope*slope-a[0]-b[0]),y=mod(slope*(a[0]-x)-a[1]);
  assert.equal(mod(y*y-x*x*x-7n),0n);return [x,y];
}
function toyRows(initial,step,ne,useSwap){
  const R=initial.length;let rx=initial.map(v=>v[0]),ry=initial.map(v=>v[1]),nx=new Array(R),ny=new Array(R);
  const out=[],stats={doublings:0,inverseErrors:0};let status=0,done=0;
  while(true){
    for(let i=0;i<Math.min(R,ne-done);i++)out.push([rx[i],ry[i]]);
    done+=Math.min(R,ne-done);if(done>=ne)break;
    const next=[];for(let i=0;i<R;i++){const v=toyAdd([rx[i],ry[i]],step,stats);if(v===null){status=-1;break;}next.push(v);}
    if(status)break;
    for(let i=0;i<R;i++){nx[i]=next[i][0];ny[i]=next[i][1];}
    if(useSwap){[rx,nx]=[nx,rx];[ry,ny]=[ny,ry];}
    else for(let i=0;i<R;i++){rx[i]=nx[i];ry[i]=ny[i];}
  }
  return {out,status,done,stats};
}
const G=[47n,71n],twoG=toyAdd(G,G,{doublings:0,inverseErrors:0});
const toyCases=[{name:'doubling in successful next row',initial:[G,twoG],step:twoG,ne:8,status:0},{name:'inverse point error in next row',initial:[G,[twoG[0],mod(-twoG[1])]],step:twoG,ne:4,status:-1},{name:'zero-y doubling error',initial:[[6n,0n]],step:[6n,0n],ne:2,status:-1},{name:'inverse point error after successful rows',initial:[G,twoG],step:twoG,ne:24,status:-1}];
const toyResults=[];
for(const c of toyCases){const a=toyRows(c.initial,c.step,c.ne,false),b=toyRows(c.initial,c.step,c.ne,true);assert.deepEqual(a,b);assert.equal(a.status,c.status);toyResults.push({name:c.name,status:a.status,emitted:a.done,doublings:a.stats.doublings,inverseErrors:a.stats.inverseErrors});}
const result={scope:'independent synthetic and toy-curve CPU models only',sourcePath,sourceDerivedConstants:{TB_R:rowSize,QSB_CG_SEG:segmentSize},configurations,segmentCalls,emittedPoints,bufferAndOutputEquivalence:true,callerAllocationIdentitiesPreserved:true,pairwiseDisjointRegions:9,errorReturnEquivalence:true,toyCurveCases:toyResults,fullSegmentCopyBytesEliminated:expectedCopyBytes,candidateExecuted:false,challengeEcArithmeticVerified:false,cppCompiled:false,gpuVerified:false,performanceMeasured:false};
console.log(JSON.stringify(result,null,2));
if (process.argv[2]) fs.writeFileSync(process.argv[2],JSON.stringify(result,null,2)+'\n');
