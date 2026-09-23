#!/usr/bin/env python3
"""Count the rolled point-chain loop, not all cold/static instructions."""
from pathlib import Path
import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--mode', choices=('native', 'jit'), default='native')
args = parser.parse_args()
root = args.directory.resolve()
results = {}
for d in sorted(root.iterdir()):
    if not d.is_dir() or not (d / f'{args.mode}.sass').exists():
        continue
    s = (d / f'{args.mode}.sass').read_text()
    fn = re.search(r'Function : (_Z23kernel_pinning_pipelineILb1ELi0[^\n]+)(.*?)(?=Function :|\Z)', s, re.S).group(2)
    ins = {int(a, 16): b for a, b in re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]*);', fn) if not b.startswith('NOP')}
    loops = []
    for a, b in ins.items():
        q = re.search(r'BRA(?:\.U)? (0x[0-9a-f]+)', b)
        if q and int(q.group(1), 16) < a:
            dest = int(q.group(1), 16)
            span = [v for p, v in ins.items() if dest <= p <= a]
            if len(span) > 500:
                loops.append({'start': hex(dest), 'end': hex(a), 'instructions': len(span), 'wide_mads': sum('IMAD.WIDE' in v for v in span)})
    log = (d / f'{args.mode}.log').read_text()
    m = re.search(r'Function properties for _Z23kernel_pinning_pipelineILb1ELi0[^\n]+\n(.*?)Compile time', log, re.S)
    results[d.name] = {'loops': loops, 'total': len(ins), 'resource': m.group(1).strip() if m else '?'}
(root / f'{args.mode}_results.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
