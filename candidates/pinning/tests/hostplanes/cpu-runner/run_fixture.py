#!/usr/bin/env python3
"""CPU-only H0/word-plane correctness and optional record-service benchmark.

Build the shared object separately from this runner; it does not compile code.
The shared-object source is the exact A/C host fixture plus service_bench.cpp.
"""
import argparse
import ctypes as C
import ctypes.util
import json
import os
import random
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXTURES = json.loads((ROOT / 'host-oracle-fixtures.json').read_text())['fixtures']
LANES = 128
REC_BYTES = LANES * 68
REC_WORDS = REC_BYTES // 4
MASK32 = (1 << 32) - 1

class Timing(C.Structure):
    _fields_ = [('cpu_ns', C.c_uint64), ('wall_ns', C.c_uint64), ('reported_hits', C.c_uint64)]

def x_limbs(rec, lane, ri):
    words = struct.unpack_from('<2176I', rec)
    out=[]
    for k in range(4):
        ix=((2*ri+k//2)*LANES+lane)*4+(k%2)*2
        out.append(words[ix] | (words[ix+1] << 32))
    return out

def yp_at(rec, lane):
    return struct.unpack_from('<I', rec, 64*LANES+4*lane)[0]

def h0(x, prefix):
    key=bytes([prefix & 255])+int(x).to_bytes(32,'big')
    keybuf=C.create_string_buffer(key)
    out=(C.c_ubyte*32)()
    if not OPENSSL_SHA256(keybuf,len(key),out): raise RuntimeError('OpenSSL SHA256 failed')
    return int.from_bytes(bytes(out[:4]), 'big')

def words_for(x, prefix):
    msg=bytes([prefix & 255])+int(x).to_bytes(32,'big')+b'\x80\0\0'
    return [int.from_bytes(msg[i:i+4], 'big') for i in range(0,36,4)]

def make_record(raw_words):
    return bytearray(struct.pack('<%dI'%REC_WORDS, *raw_words))

def set_y(rec, lane, y):
    struct.pack_into('<I', rec, 64*LANES+4*lane, y & MASK32)

def poison_x(rec, lane):
    # Candidate must not use coordinates when Y is zero or the lane is tail.
    for ri in range(2):
        for k in range(4):
            ix=((2*ri+k//2)*LANES+lane)*16+(k%2)*8
            struct.pack_into('<Q', rec, ix, 0xD15EA5E000000000 ^ (ri<<24) ^ (k<<8) ^ lane)

def create_fixture_record(i):
    f=FIXTURES[i % len(FIXTURES)]
    # These are independently generated real secp256k1 compressed-point Xs.
    return make_record(f['record_words_le32'])

def expected_record(rec, j, batch_sz, bits=24):
    base=(j*4+3)*LANES
    expected=[]
    for lane in range(LANES):
        y=yp_at(rec,lane)
        idx=base+lane
        if y==0 or idx>=batch_sz: continue
        h0v=h0(sum(v<<(64*k) for k,v in enumerate(x_limbs(rec,lane,0))),y&255)
        h1v=h0(sum(v<<(64*k) for k,v in enumerate(x_limbs(rec,lane,1))),(y>>8)&255)
        if h0v >> (32-bits)==0: expected.append(idx)
        elif h1v >> (32-bits)==0: expected.append(idx|(1<<30))
    return expected

def call_record(lib, tag, plane, j, batch, mode):
    out=(C.c_uint32*64)()
    fn=getattr(lib, 'pk_fixture_record_'+tag)
    fn.argtypes=[C.POINTER(C.c_uint8),C.c_uint32,C.c_uint32,C.c_int,C.POINTER(C.c_uint32)]
    fn.restype=C.c_uint32
    buf=(C.c_uint8*len(plane)).from_buffer(plane)
    n=fn(buf,j,batch,mode,out)
    return list(out[:min(n,64)])

def check_words_and_h0(lib, rec):
    fn=lib.pk_fixture_group_C
    fn.argtypes=[C.POINTER(C.c_uint8),C.c_int,
       C.POINTER((C.c_uint32*8)*9),C.POINTER((C.c_uint32*8)*9),
       C.POINTER(C.c_uint32*8),C.POINTER(C.c_uint32*8)]
    fn.restype=None
    buf=(C.c_uint8*len(rec)).from_buffer(rec)
    checks=0
    for l0 in range(0,LANES,8):
        w0=((C.c_uint32*8)*9)(); w1=((C.c_uint32*8)*9)()
        h0v=(C.c_uint32*8)(); h1v=(C.c_uint32*8)()
        fn(buf,l0,C.byref(w0),C.byref(w1),C.byref(h0v),C.byref(h1v))
        for t in range(8):
            lane=l0+t; y=yp_at(rec,lane)
            for ri,(ww,hh) in enumerate(((w0,h0v),(w1,h1v))):
                x=sum(v<<(64*k) for k,v in enumerate(x_limbs(rec,lane,ri)))
                want=words_for(x,(y>>(8*ri))&255)
                got=[int(ww[k][t]) for k in range(9)]
                if got != want: raise AssertionError(('word transpose',l0,lane,ri,got,want))
                digest=h0(x,(y>>(8*ri))&255)
                if int(hh[t]) != digest: raise AssertionError(('H0',l0,lane,ri,hh[t],digest))
                checks+=1
    return checks

def verify(lib, bits=24):
    if not lib.pk_fixture_has_avx2():
        raise RuntimeError('AVX2 unavailable; do not execute candidate mode')
    mode_list=[0,1]
    if lib.pk_fixture_has_shani(): mode_list.append(2)
    # Full valid records: source-extracted C plane words and H0 are checked for
    # every lane against byte serialization and OpenSSL, then record hits A/C.
    records=[create_fixture_record(i) for i in range(4)]
    fullplane=bytearray().join(records)
    fullbatch=4*4*LANES
    word_checks=0; record_checks=0
    for rec in records: word_checks+=check_words_and_h0(lib,rec)
    for mode in mode_list:
        for j,rec in enumerate(records):
            exp=expected_record(rec,j,fullbatch,bits)
            a=call_record(lib,'A',fullplane,j,fullbatch,mode)
            c=call_record(lib,'C',fullplane,j,fullbatch,mode)
            if a!=exp or c!=exp or a!=c: raise AssertionError(('full tuple',mode,j,len(a),len(c),len(exp)))
            record_checks+=1
    # Poison all inactive X words and Y markers. This catches unintended reads
    # through output mismatches for every tested tail/dead-lane combination.
    tail_checks=0
    for tail in (1,3,7,31,64,127,128):
        batch=(3*4+3)*LANES+tail
        varied=[create_fixture_record(i) for i in range(4)]
        for j,rec in enumerate(varied):
            base=(j*4+3)*LANES
            for lane in range(LANES):
                idx=base+lane
                if idx>=batch or (idx<batch and (lane%11==0)):
                    set_y(rec,lane,0 if idx<batch else 0xA5A5A5A5)
                    poison_x(rec,lane)
        variedplane=bytearray().join(varied)
        for mode in mode_list:
            for j,rec in enumerate(varied):
                exp=expected_record(rec,j,batch,bits)
                a=call_record(lib,'A',variedplane,j,batch,mode)
                c=call_record(lib,'C',variedplane,j,batch,mode)
                if a!=exp or c!=exp or a!=c: raise AssertionError(('tail tuple',tail,mode,j,a,c,exp))
                tail_checks+=1
    directed_checks=0
    if bits == 8:
        # Exercise positive publication through the actual record helper. The
        # x-coordinates are chosen by a deterministic bounded search over the
        # byte-message oracle; they test host record semantics, not EC validity.
        rng=random.Random(0x20261004)
        hit = miss = None
        for _ in range(20000):
            x=rng.getrandbits(256); p=2+(rng.getrandbits(1))
            cls=(h0(x,p)>>24)==0
            if cls and hit is None: hit=(x,p)
            if not cls and miss is None: miss=(x,p)
            if hit is not None and miss is not None: break
        if hit is None or miss is None: raise AssertionError('bounded directed hit/miss search')
        found=[(hit,miss),(miss,hit),(hit,hit)]
        rec=bytearray(REC_BYTES)
        expected=[]
        base=3*LANES
        for lane,(key0,key1) in enumerate(found):
            x0,p0=key0; x1,p1=key1
            for ri,x in ((0,x0),(1,x1)):
                # The source planes store each u64 pair at 16B/lane.
                for k in range(4):
                    addr=((2*ri+k//2)*LANES+lane)*16+(k%2)*8
                    struct.pack_into('<Q',rec,addr,(x>>(64*k))&((1<<64)-1))
            set_y(rec,lane,0x80000000|p0|(p1<<8))
            h0hit=(h0(x0,p0)>>24)==0
            h1hit=(h0(x1,p1)>>24)==0
            if h0hit: expected.append(base+lane)
            elif h1hit: expected.append((base+lane)|(1<<30))
        directed_plane=bytearray(rec)
        for mode in mode_list:
            a=call_record(lib,'A',directed_plane,0,base+3,mode)
            c=call_record(lib,'C',directed_plane,0,base+3,mode)
            if a!=expected or c!=expected: raise AssertionError(('N8 positive/priority',mode,a,c,expected))
            directed_checks+=1
    return {'status':'PASS','mode_set':mode_list,'ranked_gate_bits':bits,
            'word_and_H0_checks':word_checks,
            'record_tuple_checks':record_checks,'dead/tail tuple checks':tail_checks,
            'directed_positive_mode_checks':directed_checks,
            'tail_sizes':[1,3,7,31,64,127,128],
            'independent_oracle':'OpenSSL SHA256 over serialized 33-byte key',
            'real_point_fixture_count':len(records)}

def bench(lib, calls=2048, warm_calls=64, groups=16):
    if not lib.pk_fixture_has_avx2(): return {'status':'SKIP_no_AVX2'}
    records=[create_fixture_record(i) for i in range(4)]
    # Full records and stable initialized input. Per backend, retain 16 groups;
    # ABBA and BAAB alternate, with setup outside each timed interval.
    plane=b''.join(bytes(r) for r in records)
    buf=(C.c_uint8*len(plane)).from_buffer_copy(plane)
    fn={tag:getattr(lib,'pk_fixture_timed_record_'+tag) for tag in ('A','C')}
    for tag in fn:
        fn[tag].argtypes=[C.POINTER(C.c_uint8),C.c_uint32,C.c_uint32,C.c_int,C.c_uint32]
        fn[tag].restype=Timing
    batch=4*4*LANES
    rows=[]
    warm=[]
    modes=[1] + ([2] if lib.pk_fixture_has_shani() else [])
    for mode in modes:
        for tag in ('A','C'):
            result=fn[tag](buf,0,batch,mode,warm_calls)
            warm.append({'variant':tag,'mode':mode,'calls':warm_calls,
                         'cpu_ns':int(result.cpu_ns),'wall_ns':int(result.wall_ns),
                         'reported_hits':int(result.reported_hits)})
        for group in range(groups):
            order=('A','C','C','A') if group%2==0 else ('C','A','A','C')
            for i,tag in enumerate(order):
                j=0
                result=fn[tag](buf,j,batch,mode,calls)
                rows.append({'group':group,'order':order,'slot':i,'record':j,'variant':tag,
                  'mode':mode,'calls':calls,'cpu_ns':int(result.cpu_ns),
                  'wall_ns':int(result.wall_ns),'reported_hits':int(result.reported_hits)})
    return {'status':'COMPLETE','warmups':warm,'groups_per_mode':groups,
            'alternating_orders':['ABBA','BAAB'],'rows':rows,
            'mode_definition':{'1':'forced AVX2 A literal b59 vs C plane decoder',
                               '2':'unchanged SHA-NI A/C control when available'},
            'mode_selector_note':'production start selects SHA-NI mode2 when available; mode1 timing does not establish default CPU benefit'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--library',required=True)
    ap.add_argument('--verify',action='store_true')
    ap.add_argument('--bits',type=int,choices=(8,24),default=24)
    ap.add_argument('--bench',action='store_true')
    ap.add_argument('--calls',type=int,default=2048)
    ap.add_argument('--warm-calls',type=int,default=64)
    ap.add_argument('--groups',type=int,default=16)
    ap.add_argument('--cpu',type=int,default=None)
    args=ap.parse_args()
    if args.cpu is not None:
        allowed=os.sched_getaffinity(0)
        if args.cpu not in allowed: raise RuntimeError(f'CPU {args.cpu} outside allowed affinity {sorted(allowed)}')
        os.sched_setaffinity(0,{args.cpu})
    global OPENSSL_SHA256
    crypto_name=ctypes.util.find_library('crypto')
    if not crypto_name: raise RuntimeError('libcrypto not found')
    crypto=C.CDLL(crypto_name)
    OPENSSL_SHA256=crypto.SHA256
    OPENSSL_SHA256.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(C.c_ubyte)]
    OPENSSL_SHA256.restype=C.POINTER(C.c_ubyte)
    OPENSSL_VERSION=crypto.OpenSSL_version
    OPENSSL_VERSION.argtypes=[C.c_int]
    OPENSSL_VERSION.restype=C.c_char_p
    lib=C.CDLL(str(Path(args.library).resolve()))
    result={}
    if args.verify: result['verification']=verify(lib,args.bits)
    if args.bench: result['benchmark']=bench(lib,args.calls,args.warm_calls,args.groups)
    result['process_affinity']=sorted(os.sched_getaffinity(0))
    result['oracle']={'library':crypto_name,'version':OPENSSL_VERSION(0).decode('ascii','replace')}
    print(json.dumps(result,sort_keys=True,separators=(',',':')))
if __name__=='__main__': main()
