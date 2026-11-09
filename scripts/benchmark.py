import argparse
import json
import statistics
import time
from pathlib import Path
import torch
import numpy as np
import torch_npu
from tree_attention import KVTree, CostModel, schedule
from tree_attention.kernels import prepare, attention_matrix, pack_kv
from tree_attention.reference import request_attention


def cases():
    return {
        'no_share': KVTree.synthetic([-1]+[0]*32,[0]+[1024]*32,list(range(1,33))),
        'wide': KVTree.synthetic([-1]+[0]*32,[2048]+[64]*32,list(range(1,33))),
        'unequal': KVTree.synthetic([-1]+[0]*32,[1024]+[32+31*i for i in range(32)],list(range(1,33))),
        'deep': KVTree.synthetic([-1,0,1,1,2,2,3,3]+list(range(4,8))*4,
                                [512]*8+[64]*16,list(range(8,24))),
        'wide_long': KVTree.synthetic([-1]+[0]*128,[8128]+[64]*128,list(range(1,129))),
        'no_share_long': KVTree.synthetic([-1]+[0]*128,[0]+[8192]*128,list(range(1,129))),
    }


def timing(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.npu.synchronize()
    start=time.perf_counter()
    for _ in range(iters): fn()
    torch.npu.synchronize()
    return (time.perf_counter()-start)*1e6/iters


def vendor_inputs(q,k,v,tree,layout="BNSD"):
    maxlen=max(map(len,tree.paths))
    indices=np.zeros((len(tree.paths),maxlen),dtype=np.int64)
    for request,path in enumerate(tree.paths):
        indices[request,:len(path)]=path
    ids=torch.from_numpy(indices.ravel()).to(q.device)
    shape=(len(tree.paths),maxlen,k.shape[1],k.shape[2])
    kb=k.index_select(0,ids).view(shape)
    vb=v.index_select(0,ids).view(shape)
    if layout=="BSH":
        return q.reshape(q.shape[0],1,-1),kb.flatten(2),vb.flatten(2)
    return q[:,:,None,:].contiguous(),kb.permute(0,2,1,3).contiguous(),vb.permute(0,2,1,3).contiguous()


def vendor(q,k,v,lengths,layout='BNSD'):
    return torch_npu.npu_fused_infer_attention_score(q,k,v,num_heads=32,num_key_value_heads=8,
        input_layout=layout,scale=128**-.5,actual_seq_lengths_kv=lengths)[0].reshape(q.shape[0],32,128)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/benchmark.json')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--iters',type=int,default=30)
    parser.add_argument('--cases',nargs='*')
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--run-id',type=int,default=0)
    args=parser.parse_args()
    torch.npu.set_device(0)
    rows=[]
    for name,tree in cases().items():
        if args.cases and name not in args.cases: continue
        torch.manual_seed(args.seed)
        q=torch.randn(len(tree.paths),32,128,dtype=torch.bfloat16).npu()
        k=torch.randn(tree.storage_tokens,8,128,dtype=torch.bfloat16).npu()
        v=torch.randn_like(k)
        ref=request_attention(q.cpu(),k.cpu(),v.cpu(),tree)
        kp,vp=pack_kv(k,v)
        for method in ('request','fixed','greedy','cube','vendor','vendor_bsh'):
            compile_start=time.perf_counter()
            if method.startswith('vendor'):
                layout='BSH' if method=='vendor_bsh' else 'BNSD'
                vq,vk,vv=vendor_inputs(q,k,v,tree,layout)
                lengths=list(map(len,tree.paths))
                fn=lambda: vendor(vq,vk,vv,lengths,layout)
                plan_us=0
                setup_us=0
                jobs=0
            else:
                start=time.perf_counter()
                plan=schedule(tree,32,8,128,method)
                plan_us=(time.perf_counter()-start)*1e6
                torch.npu.synchronize()
                start=time.perf_counter()
                prepared=prepare(plan,q)
                torch.npu.synchronize()
                setup_us=(time.perf_counter()-start)*1e6
                jobs=len(prepared.jobs)
                fn=lambda: attention_matrix(q,kp,vp,prepared)
            out=fn().cpu()
            compile_ms=(time.perf_counter()-compile_start)*1e3
            torch.testing.assert_close(out,ref,atol=.008,rtol=.03)
            layout_samples=[timing(lambda: vendor_inputs(q,k,v,tree,layout) if method.startswith('vendor') else pack_kv(k,v),warmup=2,iters=5) for _ in range(args.repeats)]
            torch.npu.reset_peak_memory_stats()
            samples=[timing(fn,iters=args.iters) for _ in range(args.repeats)]
            row=dict(case=name,method=method,run_id=args.run_id,seed=args.seed,steady_us=samples,median_us=statistics.median(samples),
                     plan_us=plan_us,metadata_us=setup_us,layout_us=layout_samples,first_call_ms=compile_ms,jobs=jobs,
                     full_plan_refresh_us=plan_us+setup_us+statistics.median(layout_samples)+statistics.median(samples),
                     peak_allocated_bytes=torch.npu.max_memory_allocated(),
                     shared_tokens=tree.storage_tokens,logical_tokens=sum(map(len,tree.paths)),
                     max_abs_error=(out.float()-ref.float()).abs().max().item())
            rows.append(row)
            print(json.dumps(row),flush=True)
            Path(args.output).write_text(json.dumps(rows,indent=2))


if __name__=='__main__': main()
