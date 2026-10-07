#!/usr/bin/env python3
"""Execute the candidate's host work-credit branches against completed-slot counts."""
import os
import re
import subprocess
from pathlib import Path

arm = Path(os.environ['QSB_ARM_DIR'])
work = Path(os.environ.get('QSB_VERIFY_WORK') or os.environ.get('TMPDIR') or '/tmp') / 'qsb-hosttick-oracle'
work.mkdir(parents=True, exist_ok=True)
text = (arm / 'pinning.cu').read_text()
assignment = re.findall(r'#if !QSB_PK_ON && QSB_HOST_TICK_ACTUAL\n(.*?)#endif', text, re.S)
credits = re.findall(r'#if QSB_CPU_GRIND && QSB_HOST_GATE\n(#if QSB_PK_ON\n.*?\n#endif)\n#endif', text, re.S)
credits = [credit for credit in credits if 'qcg::tick' in credit and 'QSB_HOST_TICK_ACTUAL' in credit]
assert len(assignment) == 1 and len(credits) == 2, (len(assignment), len(credits))
source = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
static std::vector<double> credited;
namespace qcg {
static double mono_s() { return 0; }
static void tick(double, double count) { credited.push_back(count); }
}
namespace qsb_pk {
static uint64_t complete = 0;
static uint64_t completed_candidates() { return complete; }
}
static uint64_t last_completed_credit = 0;
static constexpr unsigned BATCH = 4194304;
static uint32_t slot_bsz[5] = {};
static void enqueue(unsigned s, unsigned batch_sz) {
#if QSB_PK_ON
    slot_bsz[s] = batch_sz;
#endif
@ASSIGNMENT@
}
static void drain(unsigned s) {
@DRAIN@
}
static void collect(unsigned s) {
@COLLECT@
}
int main() {
    std::mt19937 random(243581);
    for (unsigned i = 0; i < 65536; ++i) {
        unsigned s = i % 5;
        static const unsigned edges[] = {0, 1, 131071, 131072, BATCH};
        unsigned size = i < 5 ? edges[i] : random() % (BATCH + 1);
        enqueue(s, size);
        qsb_pk::complete += size;
        drain(s);
        collect(s);
        assert(credited.size() == 2);
        double expected = QSB_HOST_TICK_ACTUAL || QSB_PK_ON ? size : BATCH;
        assert(credited[0] == expected && credited[1] == expected);
        credited.clear();
    }
    puts("HOST_TICK_ORACLE PASS");
}
'''.replace('@ASSIGNMENT@', '#if !QSB_PK_ON && QSB_HOST_TICK_ACTUAL\n' + assignment[0] + '#endif').replace('@DRAIN@', credits[0]).replace('@COLLECT@', credits[1])
cpp = work / 'hosttick.cpp'
cpp.write_text(source)
for enabled in (0, 1):
    for pk in (0, 1):
        binary = work / f'hosttick-{enabled}-{pk}'
        subprocess.run(['g++', '-std=gnu++17', '-O2', '-include', 'cstdlib', f'-DQSB_HOST_TICK_ACTUAL={enabled}', f'-DQSB_PK_ON={pk}', str(cpp), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
print('HOST_TICK_ORACLE PASS 262144 cases, both drain paths, OFF and PK preservation')
