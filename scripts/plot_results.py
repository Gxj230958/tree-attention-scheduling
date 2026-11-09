import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--input',default='results/benchmark.json')
parser.add_argument('--output',default='results/attention_latency.pdf')
args=parser.parse_args()
rows=json.loads(Path(args.input).read_text())
cases=list(dict.fromkeys(r['case'] for r in rows))
methods=list(dict.fromkeys(r['method'] for r in rows))
fig,axes=plt.subplots(2,3,figsize=(12,7.5),squeeze=False)
for ax,case in zip(axes.flat,cases):
    selected={r['method']:r for r in rows if r['case']==case}
    med=[selected[m]['median_us']/1000 for m in methods]
    lower=[(selected[m]['median_us']-min(selected[m]['steady_us']))/1000 for m in methods]
    upper=[(max(selected[m]['steady_us'])-selected[m]['median_us'])/1000 for m in methods]
    refresh=[selected[m]['full_plan_refresh_us']/1000 for m in methods]
    x=np.arange(len(methods))
    ax.bar(x-.18,med,.36,label='Resident kernel',yerr=[lower,upper],capsize=2)
    ax.bar(x+.18,refresh,.36,label='Layout + plan + kernel')
    ax.set_xticks(x,methods,rotation=45,ha='right')
    ax.set_yscale('log')
    ax.set_title(case.replace('_',' '))
    ax.set_ylabel('Milliseconds')
axes[0,0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(args.output,bbox_inches='tight')
