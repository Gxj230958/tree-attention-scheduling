import argparse
import copy
import json
import time
from pathlib import Path
import torch
import torch_npu
from tree_attention import KVTree,schedule
from tree_attention.kernels import CachedDecodePlan,prepare,pack_kv,attention_matrix
from tree_attention.reference import request_attention


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=301)
    parser.add_argument('--output',default='results/decode_plan_run1.json')
    parser.add_argument('--steps',type=int,default=16)
    args=parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    base=KVTree.synthetic([-1]+[0]*32,[1024]+[64]*32,list(range(1,33)))
    prefix=base.storage_tokens
    q=torch.randn(32,32,128,dtype=torch.bfloat16).npu()
    k=torch.randn(prefix+32*args.steps,8,128,dtype=torch.bfloat16).npu()
    v=torch.randn_like(k)
    kp,vp=pack_kv(k,v)
    warm_plan=CachedDecodePlan(schedule(base,method='fixed'),q,prefix,args.steps)
    attention_matrix(q,kp,vp,warm_plan.update(1))
    torch.npu.synchronize()
    del warm_plan
    rows=[]
    for method in ('fixed','greedy','cube'):
        for mode in ('rebuild','cached'):
            tree=copy.deepcopy(base)
            leaves=list(range(1,33))
            torch.npu.synchronize()
            start=time.perf_counter()
            cached=CachedDecodePlan(schedule(tree,method=method),q,prefix,args.steps) if mode=='cached' else None
            torch.npu.synchronize()
            initial_us=(time.perf_counter()-start)*1e6
            costs=[]
            for step in range(args.steps):
                torch.npu.synchronize()
                start=time.perf_counter()
                if mode=='rebuild':
                    tree.extend_leaves(leaves,[prefix+r*args.steps+step for r in range(32)])
                    p=prepare(schedule(tree,method=method),q)
                else:
                    p=cached.update(step+1)
                torch.npu.synchronize()
                planning_us=(time.perf_counter()-start)*1e6
                attention_matrix(q,kp,vp,p)
                torch.npu.synchronize()
                # A warmup compiles a changed shape before resident timing.
                start=time.perf_counter()
                for _ in range(10):actual=attention_matrix(q,kp,vp,p)
                torch.npu.synchronize()
                kernel_us=(time.perf_counter()-start)*1e6/10
                if step in (0,args.steps-1):
                    reference_tree=copy.deepcopy(base)
                    for r in range(32):
                        reference_tree.paths[r]+=tuple(prefix+r*args.steps+t for t in range(step+1))
                    reference=request_attention(q.cpu(),k.cpu(),v.cpu(),reference_tree)
                    torch.testing.assert_close(actual.cpu(),reference,atol=.008,rtol=.03)
                costs.append({'step':step+1,'planning_us':planning_us,'kernel_us':kernel_us,'total_us':planning_us+kernel_us})
            row={'method':method,'mode':mode,'seed':args.seed,'initial_plan_us':initial_us,
                 'steps':costs,'amortized_us':(initial_us+sum(c['total_us'] for c in costs))/args.steps}
            rows.append(row)
            print(json.dumps(row),flush=True)
            Path(args.output).write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
