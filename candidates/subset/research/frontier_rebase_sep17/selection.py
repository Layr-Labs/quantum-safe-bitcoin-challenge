#!/usr/bin/env python3
"""Small offline experiment triage. Reads declared receipts; never submits or benchmarks."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

VERSION = 1
MAX_RECEIPT_AGE = 120
MIN_REQUIRED = frozenset({'source_closure','cpu_correctness','native_default',
    'native_sm89','audit_default','audit_sm89','startup_policy','package','public_note'})
KINDS = {'hypothesis':0,'cpu':1,'native':2,'gpu_controlled':3,'gpu_official':4}
HEX = re.compile(r'[0-9a-f]{64}\Z')

def stamp(value):
    d = datetime.fromisoformat(value.replace('Z','+00:00'))
    if d.tzinfo is None: raise ValueError('timestamps require timezone')
    return d.astimezone(timezone.utc)

def fresh(receipt, now):
    try:
        age = (now-stamp(receipt['observed_at'])).total_seconds()
        return 0 <= age <= MAX_RECEIPT_AGE
    except (KeyError, TypeError, ValueError, AttributeError): return False

def valid_fp(value): return isinstance(value,str) and bool(HEX.fullmatch(value))

def evidence_rank(record):
    """Ordinal evidence categories only; no metric magnitude or predicted speed scoring."""
    accepted=[]; ignored=[]
    for e in record.get('evidence',[]):
        kind=e.get('kind')
        if kind not in KINDS: raise ValueError('unknown evidence kind: '+str(kind))
        if kind=='hypothesis': continue
        if not valid_fp(record.get('source_fingerprint')) or e.get('source_fingerprint')!=record['source_fingerprint']:
            ignored.append(e.get('id','unnamed')+': source mismatch'); continue
        if e.get('status')!='pass' or not e.get('artifact'):
            ignored.append(e.get('id','unnamed')+': missing passing artifact'); continue
        if kind.startswith('gpu_') and not (e.get('measurement')=='measured' and e.get('benchmark_id')==record.get('benchmark_id') and e.get('target')==record.get('target') and e.get('comparison_id') and e.get('run_id')):
            ignored.append(e.get('id','unnamed')+': GPU measurement binding incomplete');continue
        accepted.append(kind)
    level=max((KINDS[k] for k in accepted),default=0)
    return level, sorted(set(accepted)), ignored

def qualification_reasons(record):
    q=record.get('qualification',{});required=set(q.get('required',[]))|MIN_REQUIRED
    receipts=q.get('receipts',{})
    reasons=[]
    for name in sorted(required):
        r=receipts.get(name,{})
        if not (r.get('status')=='pass' and r.get('source_fingerprint')==record.get('source_fingerprint') and valid_fp(r.get('source_fingerprint')) and r.get('artifact') and valid_fp(r.get('artifact_sha256'))):
            reasons.append('qualification:'+name)
    return reasons

def evaluate(data, now):
    if data.get('schema_version')!=VERSION: raise ValueError('unsupported schema_version')
    c=data['context'];records=data['records'];seen_ids=set();out=[]
    if not c.get('accepted_base_id') or not c.get('frontier_id'): raise ValueError('explicit accepted base and frontier required')
    pending=c.get('pending_frontier_ids',[]);heads=c.get('reviewed_pending_heads',{})
    if len(pending)!=len(set(pending)):raise ValueError('duplicate pending id')
    for r in records:
        rid=r['id']
        if rid in seen_ids:raise ValueError('duplicate record id: '+rid)
        seen_ids.add(rid)
        for key in ('base_id','frontier_id','mechanism_id','mechanism_revision','hypothesis','benchmark_id','target'):
            if not isinstance(r.get(key),str) or not r[key]:raise ValueError(rid+': missing '+key)
        correctness=r.get('correctness','unknown');compat=r.get('compatibility',{})
        if correctness not in {'pass','fail','unknown'}:raise ValueError('unknown correctness status')
        blocks=[]
        if correctness=='fail':blocks.append('known-invalid')
        for key in ('target','compiler','fixed_flags'):
            status=compat.get(key,'unknown')
            if status not in {'pass','fail','unknown'}:raise ValueError('unknown compatibility status')
            if status!='pass':blocks.append('compatibility:'+key+':'+status)
        if r['frontier_id']!=c['frontier_id']:blocks.append('frontier-changed')
        if r['base_id'] not in {c['accepted_base_id'],c.get('selected_pending_base')}:
            blocks.append('base-not-selected')
        if r['base_id']==c.get('selected_pending_base') and not (r['base_id'] in pending and heads.get(r['base_id']) and r.get('base_head')==heads[r['base_id']]):
            blocks.append('pending-head-not-reviewed')
        if r.get('source_fingerprint') in c.get('already_submitted_fingerprints',[]):blocks.append('duplicate-submitted-source')
        for f in r.get('prior_failures',[]):
            resolution=f.get('resolution',{})
            if not (f.get('resolved') is True and resolution.get('artifact') and valid_fp(r.get('source_fingerprint')) and resolution.get('source_fingerprint')==r.get('source_fingerprint')):
                blocks.append('unresolved-prior-failure:'+f.get('id','unnamed'))
        level,kinds,ignored=evidence_rank(r)
        q=qualification_reasons(r)
        out.append({'id':rid,'research_eligible':not blocks,'filter_reasons':blocks,
                    'evidence_level':level,'evidence_kinds':kinds,'ignored_evidence':ignored,
                    'measured_gpu_performance':any(k.startswith('gpu_') for k in kinds),
                    'qualification_missing':q,'correctness':correctness,'submission_allowed':False})
    # Choose evidence-backed representative; identical source OR mechanism/base/revision is duplicate.
    order=sorted(range(len(records)),key=lambda i:(bool(out[i]['filter_reasons']),-out[i]['evidence_level'],len(out[i]['qualification_missing']),records[i]['id']))
    source_seen={};mechanism_seen={}
    for i in order:
        r=records[i];v=out[i]
        if not v['research_eligible']:continue
        fp=r.get('source_fingerprint');key=(r['base_id'],r['mechanism_id'],r['mechanism_revision'])
        duplicate=source_seen.get(fp) if valid_fp(fp) else None
        duplicate=duplicate or mechanism_seen.get(key)
        if duplicate:
            v['filter_reasons'].append('duplicate-of:'+duplicate);v['research_eligible']=False
        else:
            mechanism_seen[key]=r['id']
            if valid_fp(fp):source_seen[fp]=r['id']
    for r,v in zip(records,out):
        gate=list(v['filter_reasons'])+v['qualification_missing']
        if r.get('correctness')!='pass':gate.append('correctness-not-qualified')
        if not valid_fp(r.get('source_fingerprint')):gate.append('missing-source-fingerprint')
        if r.get('source_fingerprint')!=c.get('current_source_fingerprint') or not valid_fp(c.get('current_source_fingerprint')):gate.append('selected-source-mismatch')
        fr=c.get('frontier_receipt') or {};slot=c.get('own_slot') or {}
        if not fresh(fr,now):gate.append('frontier-receipt-not-fresh')
        if not (fr.get('id') and fr.get('frontier_id')==c['frontier_id'] and fr.get('accepted_base_id')==c['accepted_base_id'] and fr.get('pending_frontier_ids')==pending and fr.get('reviewed_pending_heads')==heads):gate.append('frontier-receipt-mismatch')
        if not fresh(slot,now):gate.append('slot-receipt-not-fresh')
        if slot.get('status')!='free':gate.append('own-slot-not-free')
        if not (fr.get('id') and slot.get('frontier_receipt_id')==fr['id']):gate.append('slot-snapshot-mismatch')
        if c.get('benchmark_id')!=r['benchmark_id'] or c.get('target')!=r['target']:gate.append('target-or-benchmark-mismatch')
        v['submission_blockers']=sorted(set(gate));v['submission_allowed']=not gate
    ranked=sorted((v for v in out if v['research_eligible']),key=lambda v:(not v['submission_allowed'],-v['evidence_level'],len(v['qualification_missing']),v['id']))
    return {'schema_version':VERSION,'evaluated_at':now.isoformat(),'freshness_limit_seconds':MAX_RECEIPT_AGE,'accepted_base_id':c['accepted_base_id'],'selected_pending_base':c.get('selected_pending_base'),'pending_frontier_ids':pending,'ranked_ids':[v['id'] for v in ranked],'submission_eligible_ids':[v['id'] for v in ranked if v['submission_allowed']],'records':out,'limits':'Offline declared-receipt triage, not artifact verification, prediction of speed, GPU execution or submission authorization. GPU absence alone does not block; freshness must be rechecked at actual submission.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('records',type=Path);p.add_argument('--now',help='UTC/offset timestamp; defaults to current UTC');p.add_argument('--output',type=Path);a=p.parse_args()
    raw=a.records.read_bytes();data=json.loads(raw)
    result=evaluate(data,stamp(a.now) if a.now else datetime.now(timezone.utc));result['input_sha256']=hashlib.sha256(raw).hexdigest()
    text=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(text)
    else:print(text,end='')
if __name__=='__main__':main()
