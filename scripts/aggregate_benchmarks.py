import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('inputs',nargs='+')
parser.add_argument('--output',default='results/benchmark.json')
args=parser.parse_args()
groups=defaultdict(list)
for name in args.inputs:
    for row in json.loads(Path(name).read_text()):
        groups[(row['case'],row['method'])].append(row)
result=[]
for (case,method),rows in groups.items():
    times=[t for r in rows for t in r['steady_us']]
    layouts=[t for r in rows for t in r['layout_us']]
    row=dict(rows[0])
    row.update(steady_us=times,median_us=statistics.median(times),layout_us=layouts,
               plan_us=statistics.median(r['plan_us'] for r in rows),
               metadata_us=statistics.median(r['metadata_us'] for r in rows),
               full_plan_refresh_us=statistics.median(r['full_plan_refresh_us'] for r in rows),
               full_refresh_samples_us=[r['full_plan_refresh_us'] for r in rows],
               run_ids=[r['run_id'] for r in rows],seeds=[r['seed'] for r in rows],
               first_call_ms=[r['first_call_ms'] for r in rows],
               max_abs_error=max(r['max_abs_error'] for r in rows),
               peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows))
    row.pop('run_id');row.pop('seed')
    result.append(row)
Path(args.output).write_text(json.dumps(result,indent=2))
