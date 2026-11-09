import json
import torch
import torch_npu
import triton
from tree_attention import KVTree,schedule
from tree_attention.kernels import prepare,attention_matrix,pack_kv,_segment,_merge

torch.npu.set_device(0)
torch.manual_seed(101)
tree=KVTree.synthetic([-1]+[0]*32,[0]+[1024]*32,list(range(1,33)))
q=torch.randn(32,32,128,dtype=torch.bfloat16).npu()
k=torch.randn(tree.storage_tokens,8,128,dtype=torch.bfloat16).npu()
v=torch.randn_like(k)
p=prepare(schedule(tree),q)
kp,vp=pack_kv(k,v)
qp=q.reshape(32,8,4,128).permute(1,0,2,3).contiguous()
segment=lambda: _segment[(p.qids.shape[0],8,triton.cdiv(p.qids.shape[1]*4,16))](
    qp,kp,vp,p.qids,p.kids,p.qlen,p.klen,p.part,p.lse,
    32,8,128,p.qids.shape[1],p.kids.shape[1],32,kp.shape[2],16,128,128,p.contiguous_k,p.contiguous_q,multibuffer=False)
merge=lambda: _merge[(32,32)](p.part,p.lse,p.slots,p.counts,p.out,32,8,128,p.qids.shape[1],p.slots.shape[1],128)
for fn in [segment,merge]:
    for _ in range(3):fn()
torch.npu.synchronize()
rows=[]
for name,fn in [('segment',segment),('merge',merge)]:
    start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(20):fn()
    end.record()
    end.synchronize()
    rows.append({'kernel':name,'device_us':start.elapsed_time(end)*1000/20})
print(json.dumps(rows),flush=True)
