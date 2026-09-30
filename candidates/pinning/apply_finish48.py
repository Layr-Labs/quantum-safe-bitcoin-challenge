"""Insert one finish-only PTX register budget; preserve all other PTX bytes."""
from pathlib import Path
import hashlib
import json
import re
import sys

source = Path(sys.argv[1]).read_text()
matches = list(re.finditer(r'\.visible \.entry (_Z23kernel_pinning_pipelineILb1ELi2EE[^\s(]*)\(', source))
assert len(matches) == 1
m = matches[0]
body = source.index('\n{', m.end())
header = source[m.start():body]
assert re.search(r'\.maxntid\s+256,\s*1,\s*1', header)
assert re.search(r'\.minnctapersm\s+4', header)
assert '.maxnreg' not in header
directive = '\n.maxnreg 48'
patched = source[:body] + directive + source[body:]
assert patched[:body] + patched[body+len(directive):] == source
Path(sys.argv[2]).write_text(patched)
Path(sys.argv[3]).write_text(json.dumps({
    'function':m[1], 'maxntid':256, 'minnctapersm':4, 'maxnreg':48,
    'single_directive_added':True, 'all_other_ptx_bytes_unchanged':True,
    'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
    'patched_sha256':hashlib.sha256(patched.encode()).hexdigest(),
}, indent=2)+'\n')
