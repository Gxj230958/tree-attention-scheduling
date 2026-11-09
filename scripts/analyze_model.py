import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('inputs',nargs='+')
    parser.add_argument('--output',default='results/qwen4b.json')
    args=parser.parse_args()
    groups=defaultdict(list)
    for filename in args.inputs:
        for row in json.loads(Path(filename).read_text())['results']:
            groups[(row['workload'],row['method'],row['planning'])].append(row)
    rows=[]
    for (workload,method,planning),values in groups.items():
        row=dict(workload=workload,method=method,planning=planning,seeds=[v['seed'] for v in values])
        for metric in ('prefill_ms','decode_total_ms','generation_total_ms','decode_tokens_per_second',
                       'peak_allocated_bytes','token_agreement_with_request','physical_kv_tokens',
                       'logical_kv_tokens','allocated_kv_tokens'):
            samples=[v[metric] for v in values]
            row[metric]=median(samples)
            row[metric+'_range']=[min(samples),max(samples)]
        rows.append(row)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    workloads=list(dict.fromkeys(r['workload'] for r in rows))
    for axis,metric in zip(axes,('decode_total_ms','generation_total_ms')):
        for offset,(method,planning) in enumerate((('request','cached'),('fixed','cached'),
                                                 ('greedy','cached'),('cube','cached'),('cube','rebuild'))):
            points=[next(r for r in rows if r['workload']==w and r['method']==method and r['planning']==planning) for w in workloads]
            x=[i+(offset-2)*.16 for i in range(len(workloads))]
            center=[r[metric] for r in points]
            lower=[r[metric]-r[metric+'_range'][0] for r in points]
            upper=[r[metric+'_range'][1]-r[metric] for r in points]
            axis.bar(x,center,width=.15,label=f'{method}, {planning}',yerr=[lower,upper],capsize=2)
        axis.set_xticks(range(len(workloads)),[w.replace('_','\n') for w in workloads])
        axis.set_ylabel('Milliseconds')
        axis.set_title('Decode' if metric=='decode_total_ms' else 'Prefill + decode')
    axes[0].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(args.output).with_suffix('.pdf'))
    Path(args.output).write_text(json.dumps({'inputs':args.inputs,'results':rows},indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
