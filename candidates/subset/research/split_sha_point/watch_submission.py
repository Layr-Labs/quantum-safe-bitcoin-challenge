"""Read-only monitor for the already submitted three-stage candidate.

This never submits, cancels, edits source, or declares goal completion.
A CLI timeout is an observation error, not a terminal remote verdict.
"""
from pathlib import Path
import datetime
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SUBMISSION = 'fd16dfa4-4db2-4aee-ab1f-519c564a53b8'
JOB = '6e4f4ffa-61b6-4911-b08c-f949b6a51233'
WRAPPER = '/mnt/d/kongtaoxing/ops/qsb-yukon.sh'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, data):
    target = HERE / name
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(target)


save('watch-process.json', {'pid': os.getpid(), 'startedAt': now(),
                           'submissionId': SUBMISSION, 'pollSeconds': 120,
                           'state': 'running'})
while True:
    record = {'checkedAt': now(), 'submissionId': SUBMISSION}
    terminal = False
    try:
        result = subprocess.run(['bash', WRAPPER, 'submissions', '--json'],
                                cwd=ROOT, capture_output=True, text=True, timeout=90)
        if result.returncode:
            record.update(observationError='CLI returned nonzero', exitCode=result.returncode)
        else:
            payload = json.loads(result.stdout)
            rows = [s for s in payload['submissions'] if s['id'] == SUBMISSION]
            if not rows:
                record['observationError'] = 'Submission absent from returned list; remote outcome unknown'
            else:
                row = rows[0]
                record.update({k: row.get(k) for k in (
                    'status', 'officialScore', 'officialMetrics', 'improved',
                    'rejectionReason', 'submissionCommitSha', 'promotionStatus',
                    'promotionReason', 'promotedSourceRef', 'promotionFinishedAt')})
                record.update(jobId=JOB, priorFrontier=623518629,
                              currentFrontier=payload['benchmark']['currentBestScore'],
                              goalComplete=False)
                save('watch-latest.json', record)
                target = HERE.parent / 'split_pipeline' / 'official-status.json'
                target.write_text(json.dumps(record, indent=2) + '\n')
                terminal = row.get('status') in ('accepted', 'rejected', 'failed', 'cancelled', 'canceled')
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, OSError) as error:
        record['observationError'] = type(error).__name__
    with (HERE / 'watch-observations.jsonl').open('a') as log:
        log.write(json.dumps(record) + '\n')
    print(json.dumps({k: record.get(k) for k in ('checkedAt', 'status', 'officialScore', 'observationError')}), flush=True)
    if terminal:
        save('watch-process.json', {'pid': os.getpid(), 'finishedAt': now(),
                                   'submissionId': SUBMISSION, 'state': 'terminal remote verdict observed',
                                   'remoteStatus': record['status'], 'goalComplete': False})
        break
    time.sleep(120)
