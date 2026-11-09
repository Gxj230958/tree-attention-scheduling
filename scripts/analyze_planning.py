import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['tree','decode'])
    parser.add_argument('inputs',nargs='+')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    runs=[json.loads(Path(p).read_text()) for p in args.inputs]
    groups=defaultdict(list)
    for run in runs:
        for row in run:
            key=(json.dumps(row['shape'],sort_keys=True),row['method']) if args.kind=='tree' else (row['method'],row['mode'])
            groups[key].append(row)
    rows=[]
    for key,values in groups.items():
        if args.kind=='tree':
            row={'shape':json.loads(key[0]),'method':key[1],'requests':values[0]['requests']}
            metrics=('kernel_us','plan_us','metadata_us','refresh_us','estimated_us')
        else:
            row={'method':key[0],'mode':key[1]}
            for value in values:
                value['planning_us']=sum(s['planning_us'] for s in value['steps'])/len(value['steps'])
                value['kernel_us']=sum(s['kernel_us'] for s in value['steps'])/len(value['steps'])
            metrics=('initial_plan_us','planning_us','kernel_us','amortized_us')
        row['seeds']=[v['seed'] for v in values]
        for metric in metrics:
            row[metric]=median(v[metric] for v in values)
            row[metric+'_range']=[min(v[metric] for v in values),max(v[metric] for v in values)]
        rows.append(row)
    output={'inputs':args.inputs,'results':rows}
    if args.kind=='tree':
        shapes=defaultdict(dict)
        for row in rows:shapes[json.dumps(row['shape'],sort_keys=True)][row['method']]=row
        comparisons={}
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for axis,metric in zip(axes,('kernel_us','refresh_us')):
            points=[]
            for methods in shapes.values():
                cube=methods['cube']
                points.append((cube['requests'],methods['fixed'][metric]/cube[metric],
                               methods['greedy'][metric]/cube[metric]))
            for requests in (4,12,24):
                subset=[p for p in points if p[0]==requests]
                axis.scatter([p[1] for p in subset],[p[2] for p in subset],s=20,alpha=.65,label=f'{requests} requests')
            axis.axhline(1,color='black',linewidth=.7)
            axis.axvline(1,color='black',linewidth=.7)
            axis.set_xlabel('Fixed / dynamic latency')
            axis.set_ylabel('Greedy / dynamic latency')
            axis.set_title('Resident attention' if metric=='kernel_us' else 'Plan + metadata + attention')
            for baseline in ('request','fixed','greedy'):
                ratios=[m[baseline][metric]/m['cube'][metric] for m in shapes.values()]
                comparisons[f'{metric}_vs_{baseline}']={'geometric_mean_speedup':math.exp(sum(map(math.log,ratios))/len(ratios)),
                    'median_speedup':median(ratios),'range':[min(ratios),max(ratios)],
                    'faster_shapes':sum(x>1 for x in ratios),'shapes':len(ratios)}
        axes[0].legend(frameon=False)
        output['comparisons']=comparisons
    else:
        fig,axis=plt.subplots(figsize=(7,3.8))
        labels=[r['method']+'\n'+r['mode'] for r in rows]
        planning=[r['planning_us']+r['initial_plan_us']/len(groups[(r['method'],r['mode'])][0]['steps']) for r in rows]
        kernels=[r['kernel_us'] for r in rows]
        axis.bar(labels,planning,label='Plan/update incl. initial amortization')
        axis.bar(labels,kernels,bottom=planning,label='Resident attention')
        axis.set_ylabel('Microseconds per decode step')
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(args.output).with_suffix('.pdf'))
    Path(args.output).write_text(json.dumps(output,indent=2))
    print(json.dumps(output.get('comparisons',rows),indent=2))


if __name__=='__main__':main()
