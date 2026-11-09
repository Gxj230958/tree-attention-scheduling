import argparse
import json
import time
from pathlib import Path
import torch
import torch_npu
from tree_attention import schedule
from tree_attention.kernels import prepare,pack_kv,attention_matrix
from tree_attention.reference import request_attention
from tree_shapes import evaluation_trees


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=201)
    parser.add_argument('--output',default='results/tree_grid_run1.json')
    parser.add_argument('--iters',type=int,default=10)
    args=parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    queries={n:torch.randn(n,32,128,dtype=torch.bfloat16).npu() for n in (4,12,24)}
    k=torch.randn(65536,8,128,dtype=torch.bfloat16).npu()
    v=torch.randn_like(k)
    kp,vp=pack_kv(k,v)
    kc,vc=k.cpu(),v.cpu()
    rows=[]
    for parameters,tree in evaluation_trees():
        q=queries[len(tree.paths)]
        reference=request_attention(q.cpu(),kc,vc,tree)
        for method in ('request','fixed','greedy','cube'):
            start=time.perf_counter()
            plan=schedule(tree,method=method)
            plan_us=(time.perf_counter()-start)*1e6
            plan.validate(tree)
            torch.npu.synchronize()
            start=time.perf_counter()
            p=prepare(plan,q)
            torch.npu.synchronize()
            metadata_us=(time.perf_counter()-start)*1e6
            actual=attention_matrix(q,kp,vp,p).cpu()
            torch.testing.assert_close(actual,reference,atol=.008,rtol=.03)
            for _ in range(3):attention_matrix(q,kp,vp,p)
            torch.npu.synchronize()
            start=time.perf_counter()
            for _ in range(args.iters):attention_matrix(q,kp,vp,p)
            torch.npu.synchronize()
            kernel_us=(time.perf_counter()-start)*1e6/args.iters
            row={'shape':parameters,'requests':len(tree.paths),'storage_tokens':tree.storage_tokens,
                 'method':method,'seed':args.seed,'kernel_us':kernel_us,
                 'plan_us':plan_us,'metadata_us':metadata_us,'refresh_us':kernel_us+plan_us+metadata_us,
                 'jobs':len(p.jobs),'logical_jobs':len(plan.jobs),'estimated_us':plan.estimated_us,
                 'max_abs_error':(actual.float()-reference.float()).abs().max().item()}
            rows.append(row)
            print(json.dumps(row),flush=True)
        Path(args.output).write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
