#!/usr/bin/env python3
"""An explicit table-byte break-even model, with no claimed GPU hit rates."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
prepared=json.loads((HERE/'prepared-source.json').read_text())
assert prepared['geometry_bits']==[18]+[17]*11+[26,25]
rows=[]
for control_hit in [0,.25,.5,.6,.7,.75,.8,.85,.8666666666666667,.9,.95,1]:
 for cold_hit in [0,.05]:
  threshold=(15*control_hit-1-2*cold_hit)/12
  rows.append({'assumed_control_hit_rate':control_hit,'assumed_cold_hit_rate':cold_hit,
   'minimum_small_table_hit_rate_for_no_extra_table_bytes':max(0,threshold),
   'achievable_in_model':threshold<=1,
   'control_miss_bytes_per_candidate':64*15*(1-control_hit),
   'asymmetric_miss_bytes_with_perfect_small_hits':64*2*(1-cold_hit)})
report={'source_fingerprint':prepared['candidate_source']['source_fingerprint'],
 'status':'SYMBOLIC_COST_SCREEN','gpu_executed':False,
 'equations':{'control_bytes':'64*15*(1-h_control)',
 'asymmetric_bytes':'64*(12*(1-h_small)+2*(1-h_cold))',
 'no_extra_table_bytes':'h_small >= (15*h_control - 1 - 2*h_cold)/12'},
 'assumptions':['Uniform64Bpoint requests; hit rates are hypothetical request-weighted inputs, not measured probabilities.',
 'Counts logical table bytes missed; omits prefetch duplication, cache-line traffic amplification, writeback/checkpoint interference and overlapping requests.',
 'Both variants have the same319.5B/candidate logical main checkpoint roundtrip; its DRAM traffic need not be equal because cache policies interact.',
 'Byte parity does not imply time parity:14serialpoint dependencies differ from15, and saved7M2S may offset more bytes or fail to offset latency.',
 '3.05GiBtable must be generated at runtime; startup/setup and warmup time remain unmeasured.'],
 'scenarios':rows,
 'decision':'CPU/native checks preserve plausibility, but cache reuse is the deciding unknown. No supported substantial expected-lead claim over current/pending submissions yet.'}
(HERE/'cache-budget.json').write_text(json.dumps(report,indent=2)+'\n')
print('At control hit80%, cold hit0%, small tables need91.67% hits for byte parity; above86.67% control hits, even perfect small hits add table bytes. These are assumptions, not measurements.')
