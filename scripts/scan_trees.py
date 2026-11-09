import json
from itertools import product
from pathlib import Path
from tree_attention import KVTree,schedule

rows=[]
for root,child,left,right in product((16,128,1024),(8,64,256),(16,128,512),(16,128,512)):
    lengths=[root,child,child,left,right,left,right]
    tree=KVTree.synthetic([-1,0,0,1,1,2,2],lengths,[3,4,5,6])
    plans={m:schedule(tree,method=m) for m in ('request','fixed','greedy','cube')}
    normalized={m:sorted((j.requests,j.tokens) for j in p.jobs) for m,p in plans.items()}
    rows.append({'lengths':lengths,'cost_us':{m:p.estimated_us for m,p in plans.items()},
                 'jobs':{m:len(p.jobs) for m,p in plans.items()},
                 'cube_differs_fixed':normalized['cube']!=normalized['fixed'],
                 'cube_differs_greedy':normalized['cube']!=normalized['greedy']})
Path('results/tree_scan.json').write_text(json.dumps(rows,indent=2))
print(json.dumps({'trees':len(rows),'differs_fixed':sum(r['cube_differs_fixed'] for r in rows),
                  'differs_greedy':sum(r['cube_differs_greedy'] for r in rows)},indent=2))
print(json.dumps([r for r in rows if r['cube_differs_greedy']][:5],indent=2))
