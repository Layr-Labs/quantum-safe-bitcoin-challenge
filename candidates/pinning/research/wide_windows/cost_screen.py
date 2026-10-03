#!/usr/bin/env python3
"""Arithmetic/byte budget only; never a performance prediction or GPU benchmark."""
import json
from pathlib import Path
FRONTIER=644546620
TARGET_FACTOR=1.25
batch=1<<24
blocks=batch//256
groups=(blocks+255)//256
checkpoint_bytes={'state':batch*8*16,'tree':blocks*4*256*8,
                  'roots':blocks*32,'root_checkpoint':groups*4*256*8,'super_roots':groups*32}
# Logical loads/stores, not allocation sizes or measured DRAM transactions.
# Search tree stores/loads254 nodes; pad slots255/256 are never transferred.
checkpoint_traffic={'state_read_write':batch*128*2,
 'tree_read_write':blocks*254*32*2,
 'roots_five_transfers':blocks*32*5,
 'root_tree_read_write':groups*254*32*2,
 'super_roots_four_transfers':groups*32*4}
checkpoint_per_candidate=sum(checkpoint_traffic.values())/batch
rows=[]
for windows in (15,14,13,12,11,10):
    low,extra=divmod(256,windows)
    widths=[low+1]*extra+[low]*(windows-extra)
    table=sum(1<<(b-1) for b in widths)*64
    # Seed3M2S + (w-3) deferred7M2S + final8M2S.
    mul=3+7*(windows-3)+8;sqr=2*(windows-1)
    products=mul*73+sqr*45
    rows.append({'windows':windows,'widths':widths,'table_bytes':table,
      'table_gib':table/(1<<30),'multiplications':mul,'squares':sqr,
      'wide_products_proxy':products,'field_product_reduction_from_15':1-products/(95*73+28*45),
      'lookup_bytes_per_candidate':windows*64,
      'logical_lookup_gb_per_s_at_target':windows*64*FRONTIER*TARGET_FACTOR/1e9,
      'checkpoint_bytes_per_candidate':checkpoint_per_candidate,
      'total_logical_bytes_per_candidate_split':windows*64+checkpoint_per_candidate,
      'total_logical_gb_per_s_at_target_split':(windows*64+checkpoint_per_candidate)*FRONTIER*TARGET_FACTOR/1e9,
      'search_allocations_gib_before_misc':(table+sum(checkpoint_bytes.values()))/(1<<30)})
# This is a sensitivity calculation: F is baseline time fraction attributable
# to the optimized field chain. It is intentionally supplied, not measured.
reduction=rows[-1]['field_product_reduction_from_15']
scenarios=[]
for field_fraction in (.5,.65,.8,.9):
    remaining=1-field_fraction*reduction
    scenarios.append({'assumed_point_chain_time_fraction':field_fraction,
       'ideal_speedup_if_time_scaled_with_products':1/remaining,
       'extra_time_budget_fraction_for_25pct_gain':1/TARGET_FACTOR-remaining})
out={'kind':'unmeasured arithmetic and byte budget','frontier_score':FRONTIER,'existing_search_checkpoint_bytes':checkpoint_bytes,
 'logical_checkpoint_traffic_per_full_batch':checkpoint_traffic,
 'fused_tradeoff':{'logical_checkpoint_bytes_per_candidate':0,'ten_window_table_bytes_per_candidate':640,
   'scalar_inverses_per_full_batch':blocks,'split_scalar_inverses_per_full_batch':1,
   'limits':'Fused search still uses shared-memory trees and has local stack accesses in the root-inverse lane. Eliminating global checkpoints is not eliminating all memory traffic.'},
 'target_factor':TARGET_FACTOR,'rows':rows,'ten_window_scenarios':scenarios,
 'limits':'Wide-products proxy excludes additions, recoding, scheduling, memory hierarchy, table construction, hashing, recovery and inverse costs. Time scenarios assume product-count scaling and are not measured. Logical bytes include search checkpoints but are not physical DRAM traffic or a bandwidth guarantee; do not equate the cache-resident15-window traffic with16GiB-table misses. Allocation is live search table plus existing16M checkpoint buffers before runtime/driver overhead.'}
HERE=Path(__file__).resolve().parent
(HERE/'cost-results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
