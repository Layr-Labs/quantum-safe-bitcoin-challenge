"""CPU check of the actual shipped per-sequence SHA helpers against OpenSSL."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


HARNESS = r'''
static uint64_t random_state=0x63a48aa5e9d17239ULL;
static uint32_t word() {
 random_state^=random_state<<13; random_state^=random_state>>7; random_state^=random_state<<17;
 return uint32_t(random_state);
}
int main() {
 for(int n=0;n<32768;n++) {
  uint32_t mid[8],live[3],got[8],words[16]={};
  for(int i=0;i<8;i++)mid[i]=n<4 ? (n==0?0u:n==1?~0u:n==2?0xaaaaaaaau:0x55555555u) : word();
  for(int i=0;i<3;i++)live[i]=n<4 ? mid[0] : word();
  for(int i=0;i<3;i++)words[i]=live[i]; words[15]=79960;
  qsb_tail_pre tp; qsb_make_tail_pre(&tp,mid,live[2]);
  if(std::memcmp(tp.mid,mid,sizeof(mid)))return 9;
  std::memcpy(got,mid,sizeof(mid));
  _SHA256TransformFastTail11P(got,live[0],live[1],live[2],tp);
  SHA256_CTX ctx; SHA256_Init(&ctx);
  for(int i=0;i<8;i++)ctx.h[i]=mid[i];
  unsigned char block[64];
  for(int i=0;i<16;i++)for(int b=0;b<4;b++)block[4*i+b]=words[i]>>(24-8*b);
  SHA256_Transform(&ctx,block);
  for(int i=0;i<8;i++)if(got[i]!=ctx.h[i])return 10;
 }
 std::puts("32768 complete extracted precompute+SHA compressions matched OpenSSL");
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--include', type=Path, action='append', default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = (args.source/'pinning.cu').read_text()
    header = (args.source/'GPUHash.h').read_text()
    constants = header[header.index('__device__ __constant__ uint32_t K[]'):]
    constants = constants[:constants.index('};')+2]
    macros = '\n'.join(line for line in header.splitlines() if line.startswith(
        ('#define ROR(x,n) ', '#define S0(x) ', '#define S1(x) ', '#define s0(x) ',
         '#define s1(x) ', '#define Maj(x,y,z) ', '#define Ch(x,y,z) ')))
    rounds = header[header.index('#define S2Round('):header.index('//Take the last')]
    helpers = source[source.index('/* Per-sequence tail precompute (QSB_TAIL_PRE).'):]
    helpers = helpers[:helpers.index('/* Sparse-schedule SHA-256')]
    cpu = '#include <cstdint>\n#include <cstdio>\n#include <cstring>\n#include <openssl/sha.h>\n#define __device__\n#define __constant__\n#define __forceinline__ inline\n'
    cpu += constants+'\n'+macros+'\n'+rounds+'\n'+helpers+HARNESS
    (args.output/'check.cpp').write_text(cpu)
    flags = ['g++', '-O2', '-std=c++17', '-Wno-deprecated-declarations',
             *['-I'+str(p.resolve()) for p in args.include]]
    def compile_run(name, text, expected, sanitize=False):
        cpp = (args.output/(name+'.cpp')).resolve()
        binary = (args.output/name).resolve()
        cpp.write_text(text)
        command = flags + (['-fsanitize=undefined', '-fno-sanitize-recover=all'] if sanitize else [])
        command += [str(cpp), '-l:libcrypto.so.3', '-o', str(binary)]
        with (args.output/(name+'.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = subprocess.run([str(binary)], stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == expected, (name, result.returncode)
    compile_run('check', cpu, 0, True)
    old = 'tp->v[0] = t1p + S0a + maj;'
    assert cpu.count(old) == 1
    compile_run('negative', cpu.replace(old, 'tp->v[0] = t1p + S0a + maj + 1u;'), 10)
    proof = {'passed': True, 'cases': 32768, 'undefined_behavior_sanitizer_passed': True,
             'wrong_precomputed_round0_rejected': True,
             'source_hashes': {name: hashlib.sha256((args.source/name).read_bytes()).hexdigest()
                               for name in ('pinning.cu', 'GPUHash.h')},
             'extracted_cpu_sha256': hashlib.sha256(cpu.encode()).hexdigest(),
             'scope': 'CPU compression equivalence; native and production qualification are separate.'}
    (args.output/'proof.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
