#!/usr/bin/env python3
"""Audit the promoted composed inverse tree with the mixed-table XYZZ core."""

from __future__ import annotations

import random
import re
from pathlib import Path


P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)


def affine_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    x1, y1 = a
    x2, y2 = b
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None
        slope = 3 * x1 * x1 * pow(2 * y1, -1, P) % P
    else:
        slope = (y2 - y1) * pow(x2 - x1, -1, P) % P
    x3 = (slope * slope - x1 - x2) % P
    return x3, (slope * (x1 - x3) - y1) % P


def scalar_mult(k, point=G):
    out = None
    addend = point
    k %= N
    while k:
        if k & 1:
            out = affine_add(out, addend)
        addend = affine_add(addend, addend)
        k >>= 1
    return out


def mm(a, b):
    x1, y1 = a
    x2, y2 = b
    h = (x2 - x1) % P
    r = (y2 - y1) % P
    hh = h * h % P
    hhh = h * hh % P
    q = x1 * hh % P
    x = (r * r - hhh - 2 * q) % P
    y = (r * (q - x) - y1 * hhh) % P
    return x, y, hh, hhh


def mm_deferred(a, b):
    """D(P1+P2; y1): stored Y is actual Y + y1*ZZZ."""
    x1, y1 = a
    x2, y2 = b
    h = (x2 - x1) % P
    r = (y2 - y1) % P
    hh = h * h % P
    hhh = h * hh % P
    q = x1 * hh % P
    x = (r * r - hhh - 2 * q) % P
    yd = r * (q - x) % P
    return x, yd, hh, hhh


def madd_old(s, a):
    x, y, zz, zzz = s
    xa, ya = a
    u = xa * zz % P
    ss = ya * zzz % P
    h = (u - x) % P
    r = (ss - y) % P
    hh = h * h % P
    hhh = h * hh % P
    q = x * hh % P
    xn = (r * r - hhh - 2 * q) % P
    yn = (r * (q - xn) - y * hhh) % P
    return xn, yn, zz * hh % P, zzz * hhh % P


def madd_from_deferred(s, a, old_anchor_y, defer_output):
    """Add affine a to D(P; old_anchor_y), returning D(P+a; a.y) or exact."""
    x, yd, zz, zzz = s
    xa, ya = a
    u = xa * zz % P
    ss = (ya + old_anchor_y) * zzz % P
    h = (u - x) % P
    r = (ss - yd) % P
    hh = h * h % P
    hhh = h * hh % P
    v = u * hh % P
    xn = (r * r + hhh - 2 * v) % P
    zzn = zz * hh % P
    zzzn = zzz * hhh % P
    ycore = r * (v - xn) % P
    yn = ycore if defer_output else (ycore - ya * zzzn) % P
    return xn, yn, zzn, zzzn


def normalize(s):
    x, y, zz, zzz = s
    assert zz and zzz
    return x * pow(zz, -1, P) % P, y * pow(zzz, -1, P) % P


def denominators_nonzero(points):
    s = mm(points[0], points[1])
    if s[2] == 0:
        return False
    for a in points[2:]:
        if (a[0] * s[2] - s[0]) % P == 0:
            return False
        s = madd_old(s, a)
    return True


def check(points, check_curve):
    assert len(points) == 15
    exact = mm(points[0], points[1])
    chained = mm_deferred(points[0], points[1])
    anchor_y = points[0][1]
    assert chained[0] == exact[0] and chained[2:] == exact[2:]
    assert chained[1] == (exact[1] + anchor_y * exact[3]) % P
    for i, point in enumerate(points[2:], start=2):
        exact = madd_old(exact, point)
        final = i == len(points) - 1
        chained = madd_from_deferred(chained, point, anchor_y, not final)
        if final:
            assert chained == exact, (i, points, exact, chained)
        else:
            anchor_y = point[1]
            assert chained[0] == exact[0] and chained[2:] == exact[2:]
            assert chained[1] == (exact[1] + anchor_y * exact[3]) % P
    if check_curve:
        want = None
        for point in points:
            want = affine_add(want, point)
        assert want is not None
        assert normalize(chained) == want


def inverse_model(values):
    """Mirror PR77's full shared tree with warp-local synchronization."""
    size=len(values)
    assert size in (32,64,128,256)
    tree=[None]*(2*size)
    ops=block_barriers=warp_syncs=0
    def mul(a,b):
        nonlocal ops
        ops+=1
        return a*b%P
    for tid,value in enumerate(values):
        tree[size+tid]=value
    block_barriers+=1
    width=size//2
    while width:
        for tid in range(width):
            node=width+tid
            tree[node]=mul(tree[2*node],tree[2*node+1])
        if width>32:block_barriers+=1
        else:warp_syncs+=1
        width//=2
    tree[1]=pow(tree[1],-1,P)
    block_barriers+=1
    width=1
    while width<size:
        for tid in range(width):
            node=width+tid
            parent,left,right=tree[node],tree[2*node],tree[2*node+1]
            tree[2*node]=mul(parent,right)
            tree[2*node+1]=mul(parent,left)
        if (width<<1)>32:block_barriers+=1
        else:warp_syncs+=1
        width*=2
    assert ops==3*size-3,(size,ops)
    assert block_barriers=={32:2,64:3,128:5,256:7}[size]
    assert warp_syncs=={32:10,64:11,128:11,256:11}[size]
    return tree[size:2*size]


def recode(k):
    m=2*(k%N)%N
    sign=1 if m&1 else -1
    if sign<0:m=N-m
    out=[]
    out.append(sign*((m&((1<<19)-1))-(1<<18)))
    m=2*(m>>19)+1
    for _ in range(13):
        out.append(sign*((m&((1<<18)-1))-(1<<17)))
        m=2*(m>>18)+1
    out.append(sign*m)
    assert len(out)==15 and all(v&1 for v in out)
    assert abs(out[0]) < 1<<18
    assert all(abs(v)<1<<17 for v in out[1:])
    shifts=[0]+[17*c+1 for c in range(1,15)]
    assert sum(v<<shift for v,shift in zip(out,shifts))%N==2*k%N
    return out


def finish_homogeneous(X, Y, Z, xR, yR):
    d=(xR*Z-X)%P
    inv=pow(Z*d%P,-1,P)
    iZ=inv*d%P
    xP=X*iZ%P
    yP=Y*iZ%P
    slope1=(yR-yP)*pow(xR-xP,-1,P)%P
    slope2=-(yR+yP)*pow(xR-xP,-1,P)%P
    x1=(slope1*slope1-xP-xR)%P
    x2=(slope2*slope2-xP-xR)%P
    y1=(slope1*(xP-x1)-yP)%P
    y2=(slope2*(xP-x2)-yP)%P
    return x1,x2,y1&1,y2&1


def finish_xyzz(X, Y, ZZ, ZZZ, xR, yR):
    d=(xR*ZZ-X)%P
    W=ZZ*ZZ%P*d%P
    inv=pow(W,-1,P)
    C=ZZ*d%P*d%P
    h=ZZZ*inv%P
    delta=C*inv%P
    xs=(2*xR-delta)%P
    m1=(yR*ZZZ-Y)%P*h%P
    x1=(m1*m1-xs)%P
    y1=(m1*(xR-x1)-yR)%P
    m2=(yR*ZZZ+Y)%P*h%P
    x2=(m2*m2-xs)%P
    s2=(m2*(xR-x2)-yR)%P
    return x1,x2,y1&1,(s2&1)^1


def source_audit():
    root=Path(__file__).resolve().parent
    s=(root/'tests/gpu_epochs/tree.cu').read_text()
    m=(root/'GPUMath.h').read_text()
    t=(root/'tests/gpu_epochs/tree_inverse.cuh').read_text()
    assert 'int32_t gte[16]' not in s
    assert '_FixedBaseSignedXYZZStream(qx,qy,qzz,qzzz,z,d_gt)' in s
    assert 'gt_mixed_step<18>(M,sign)' in s
    assert 'gt_mixed_step<17>(M,sign)' in s
    assert 'for (int c=2;c<GT_CHUNKS;c++)' in s
    assert '_PointAddXYZZ_mm_def(' in s
    assert '_PointAddXYZZ_def(' in s
    assert '#define GT_CHUNKS 15' in s
    assert '#define GT_TOTAL_ENTRIES (1u << 20)' in s
    assert 'qsb_xyzz_finish_prepare(qx,qzz,u2rx,prod)' in s
    assert 'qsb_xyzz_finish_precomputed(qx,qy,Wsave,qzzz,prod' in s
    assert 'uint64_t *pts_x[2]' not in s
    assert '#include "square32.cuh"' in m
    assert '_PointAddXYZZ_def(' in m
    assert '_PointAddXYZZ_mm_def(' in m
    assert '__shared__ uint64_t tree[4][512]' in t
    assert 'const int tid=threadIdx.x,n=blockDim.x;' in t
    assert 'for(int width=n>>1;width>0;width>>=1)' in t
    assert 'for(int width=1;width<n;width<<=1)' in t
    assert 'if(width>32)__syncthreads();else __syncwarp();' in t
    assert 'if((width<<1)>32)__syncthreads();else __syncwarp();' in t
    assert '__shfl_sync' not in t
    # Resolve the entire production include closure without compiling it.
    seen=set()
    def visit(p):
        p=p.resolve()
        if p in seen:return
        assert p.is_file(),p
        assert p.is_relative_to(root.resolve()),p
        seen.add(p)
        for inc in re.findall(r'^\s*#include\s+"([^"]+)"',p.read_text(),re.M):
            visit(p.parent/inc)
    visit(root/'subset.cu')
    return len(seen)


if __name__=='__main__':
    import json
    rng=random.Random(0x260917)
    includes=source_audit()
    inverse_cases=0
    for n in (32,64,128,256):
        for run in range(40):
            vals=[rng.randrange(1,P) for _ in range(n)]
            if run<4:vals=[(1,2,P-1,P-2)[run]]*n
            if run in (4,5,6):
                active=(0,1,n-1)[run-4]
                vals[active:]=[1]*(n-active)
            got=inverse_model(vals)
            assert got==[pow(v,-1,P) for v in vals]
            inverse_cases+=n
    arbitrary=0
    while arbitrary<20000:
        pts=tuple((rng.randrange(P),rng.randrange(P)) for _ in range(15))
        if denominators_nonzero(pts):check(pts,False);arbitrary+=1
    pool=[scalar_mult(rng.randrange(1,N)) for _ in range(64)]
    curves=0
    while curves<1000:
        pts=tuple(rng.choice(pool) for _ in range(15))
        if denominators_nonzero(pts):check(pts,True);curves+=1
    scalars=[0,1,N-1,N,N+1,2**256-1]
    scalars += [1<<b for b in range(256)]
    scalars += [rng.getrandbits(256) for _ in range(10000)]
    for k in scalars:recode(k)
    # Complete mixed-table multiply and raw-XYZZ finish representation.
    # Force boundary raw hashes >=n as well as random folded runtime bases.
    recovered=0
    for k in scalars[:6]+scalars[-24:]:
        if k%N==0:continue # inherited incomplete group law has no finite sum
        nri=rng.randrange(1,N)
        base=scalar_mult(nri*pow(2,-1,N)%N)
        shifts=[0]+[17*c+1 for c in range(1,15)]
        pts=tuple(scalar_mult(e*(1<<shift),base) for e,shift in zip(recode(k),shifts))
        assert denominators_nonzero(pts)
        check(pts,True)
        exact=mm_deferred(*pts[:2]);anchor=pts[0][1]
        for i in range(2,15):
            exact=madd_from_deferred(exact,pts[i],anchor,i!=14)
            anchor=pts[i][1]
        x,y,zz,zzz=exact
        got=normalize(exact)
        assert got==scalar_mult(k*nri)
        recovered+=1
    finish_cases=0
    while finish_cases<10000:
        xP,yP,xR,yR=(rng.randrange(1,P) for _ in range(4))
        z=rng.randrange(1,P)
        zz=z*z%P; zzz=zz*z%P
        X=xP*zz%P; Y=yP*zzz%P
        if (xR*zz-X)%P==0:continue
        assert finish_xyzz(X,Y,zz,zzz,xR,yR)==finish_homogeneous(
            xP*z%P,yP*z%P,z,xR,yR
        )
        finish_cases+=1
    print(json.dumps(dict(status='PASS',include_files=includes,inverse_outputs=inverse_cases,
        arbitrary_field_chains=arbitrary,curve_chains=curves,scalar_recodes=len(scalars),
        complete_mixed_table_multiplies=recovered,xyzz_finish_cases=finish_cases,tree_shared_bytes=16384,
        block_barriers_256=7,warp_sync_phases_256=11,multiplies_256=765),indent=2))
