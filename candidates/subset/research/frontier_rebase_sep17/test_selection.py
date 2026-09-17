import copy
import unittest
from datetime import datetime, timezone, timedelta
import selection as s

NOW=datetime(2026,9,17,15,0,tzinfo=timezone.utc)
FP='a'*64

def fixture():
    r={'id':'candidate','source_fingerprint':FP,'base_id':'accepted','frontier_id':'frontier',
       'mechanism_id':'ring','mechanism_revision':'v1','hypothesis':'Reduce ring wait overhead.',
       'benchmark_id':'subset','target':'sm52-jit89','correctness':'pass',
       'compatibility':dict.fromkeys(('target','compiler','fixed_flags'),'pass'),
       'evidence':[],'prior_failures':[],
       'qualification':{'required':[], 'receipts':{k:{'status':'pass','source_fingerprint':FP,'artifact':k+'.json','artifact_sha256':'b'*64} for k in s.MIN_REQUIRED}}}
    fr={'id':'receipt1','frontier_id':'frontier','accepted_base_id':'accepted','pending_frontier_ids':[],
        'reviewed_pending_heads':{},'observed_at':NOW.isoformat()}
    c={'accepted_base_id':'accepted','selected_pending_base':None,'frontier_id':'frontier',
       'benchmark_id':'subset','target':'sm52-jit89','current_source_fingerprint':FP,
       'pending_frontier_ids':[],'reviewed_pending_heads':{},'frontier_receipt':fr,
       'own_slot':{'status':'free','frontier_receipt_id':'receipt1','observed_at':NOW.isoformat()},
       'already_submitted_fingerprints':[]}
    return {'schema_version':1,'context':c,'records':[r]}

class SelectionTests(unittest.TestCase):
    def evaluate(self,d):return s.evaluate(d,NOW)
    def row(self,d):return self.evaluate(d)['records'][0]
    def test_qualified_without_gpu_allowed(self):
        r=self.row(fixture());self.assertTrue(r['submission_allowed']);self.assertFalse(r['measured_gpu_performance'])
    def test_fresh_boundary_and_future(self):
        for seconds,want in [(120,True),(121,False),(-1,False)]:
            d=fixture();d['context']['frontier_receipt']['observed_at']=(NOW-timedelta(seconds=seconds)).isoformat()
            self.assertEqual(self.row(d)['submission_allowed'],want)
    def test_busy_slot_preserved(self):
        d=fixture();d['context']['own_slot']['status']='validating';self.assertFalse(self.row(d)['submission_allowed'])
    def test_stale_slot_or_other_snapshot(self):
        for k,v in [('observed_at',(NOW-timedelta(seconds=121)).isoformat()),('frontier_receipt_id','other')]:
            d=fixture();d['context']['own_slot'][k]=v;self.assertFalse(self.row(d)['submission_allowed'])
    def test_all_minimum_evidence_required_even_empty_required_list(self):
        for key in s.MIN_REQUIRED:
            d=fixture();del d['records'][0]['qualification']['receipts'][key]
            self.assertIn('qualification:'+key,self.row(d)['submission_blockers'])
    def test_wrong_source_receipt(self):
        d=fixture();d['records'][0]['qualification']['receipts']['native_default']['source_fingerprint']='c'*64
        self.assertFalse(self.row(d)['submission_allowed'])
    def test_invalid_or_unknown_correctness_not_submit(self):
        for status in ['fail','unknown']:
            d=fixture();d['records'][0]['correctness']=status;self.assertFalse(self.row(d)['submission_allowed'])
    def test_fixed_flags_and_compiler_filter(self):
        for key in ['fixed_flags','compiler','target']:
            d=fixture();d['records'][0]['compatibility'][key]='fail';self.assertEqual(self.evaluate(d)['ranked_ids'],[])
    def test_incompatible_proposal_no_fingerprint_supported(self):
        d=fixture();r=d['records'][0];r['source_fingerprint']=None;r['compatibility']['compiler']='fail'
        self.assertFalse(self.row(d)['research_eligible'])
    def test_static_prediction_never_gpu(self):
        d=fixture();d['records'][0]['evidence']=[{'kind':'native','measurement':'predicted','predicted_speedup':10000,'status':'pass','source_fingerprint':FP,'artifact':'sass.json'}]
        r=self.row(d);self.assertEqual(r['evidence_level'],2);self.assertFalse(r['measured_gpu_performance'])
    def test_gpu_requires_actual_binding(self):
        e={'kind':'gpu_controlled','measurement':'predicted','status':'pass','source_fingerprint':FP,'artifact':'gpu.json','benchmark_id':'subset','target':'sm52-jit89','comparison_id':'base','run_id':'run'}
        d=fixture();d['records'][0]['evidence']=[e];self.assertEqual(self.row(d)['evidence_level'],0)
        e['measurement']='measured';self.assertEqual(self.row(d)['evidence_level'],3)
        e['source_fingerprint']='c'*64;self.assertEqual(self.row(d)['evidence_level'],0)
    def test_duplicate_mechanism_and_source(self):
        for same_source in [False,True]:
            d=fixture();r=copy.deepcopy(d['records'][0]);r['id']='zcopy'
            if not same_source:r['source_fingerprint']='c'*64
            else:r['mechanism_id']='renamed'
            d['records'].append(r);result=self.evaluate(d)
            self.assertEqual(result['ranked_ids'],['candidate']);self.assertIn('duplicate-of:candidate',result['records'][1]['filter_reasons'])
    def test_prior_failure_requires_source_bound_resolution(self):
        d=fixture();f={'id':'carry-bug','resolved':True};d['records'][0]['prior_failures']=[f]
        self.assertFalse(self.row(d)['research_eligible'])
        f['resolution']={'artifact':'regression.json','source_fingerprint':FP};self.assertTrue(self.row(d)['research_eligible'])
    def test_already_submitted_source_filtered(self):
        d=fixture();d['context']['already_submitted_fingerprints']=[FP];self.assertFalse(self.row(d)['research_eligible'])
    def test_current_source_and_frontier_must_match(self):
        for key in ['current_source_fingerprint','frontier_id']:
            d=fixture();d['context'][key]='different';self.assertFalse(self.row(d)['submission_allowed'])
    def test_pending_head_review_and_snapshot(self):
        d=fixture();c=d['context'];r=d['records'][0];c['selected_pending_base']='pending';r['base_id']='pending';r['base_head']='head1';c['pending_frontier_ids']=['pending']
        self.assertFalse(self.row(d)['research_eligible'])
        c['reviewed_pending_heads']={'pending':'head1'};self.assertTrue(self.row(d)['research_eligible']);self.assertFalse(self.row(d)['submission_allowed'])
        c['frontier_receipt'].update(pending_frontier_ids=['pending'],reviewed_pending_heads={'pending':'head1'})
        self.assertTrue(self.row(d)['submission_allowed'])
    def test_schema_typos_fail_closed(self):
        d=fixture();d['records'][0]['evidence']=[{'kind':'speedup'}]
        with self.assertRaises(ValueError):self.evaluate(d)
    def test_ordinal_ranking_no_metric_bonus(self):
        d=fixture();a=d['records'][0];b=copy.deepcopy(a);b.update(id='cpu',mechanism_id='other',source_fingerprint='c'*64)
        a['evidence']=[{'kind':'native','status':'pass','source_fingerprint':FP,'artifact':'native.json','predicted_speedup':0.01}]
        b['evidence']=[{'kind':'cpu','status':'pass','source_fingerprint':'c'*64,'artifact':'cpu.json','predicted_speedup':100000}]
        d['records'].append(b);self.assertEqual(self.evaluate(d)['ranked_ids'],['candidate','cpu'])

if __name__=='__main__':unittest.main()
