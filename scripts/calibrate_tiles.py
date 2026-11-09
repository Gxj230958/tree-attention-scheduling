import json
from pathlib import Path
import torch
import torch_npu
import triton
from tree_attention import KVTree,CostModel,schedule
from tree_attention.kernels import prepare,pack_kv,_segment,_merge
from tree_attention.reference import request_attention

torch.npu.set_device(0)
torch.manual_seed(17)
rows=[]
for name,tree in [('private',KVTree.synthetic([-1]+[0]*24,[0]+[768]*24,list(range(1,25)))),
                  ('shared',KVTree.synthetic([-1]+[0]*24,[768]+[48]*24,list(range(1,25))))]:
    q=torch.randn(24,32,128,dtype=torch.bfloat16).npu()
    k=torch.randn(tree.storage_tokens,8,128,dtype=torch.bfloat16).npu()
    v=torch.randn_like(k)
    kp,vp=pack_kv(k,v)
    qp=q.reshape(24,8,4,128).permute(1,0,2,3).contiguous()
    p=prepare(schedule(tree,method='fixed'),q)
    ref=request_attention(q.cpu(),k.cpu(),v.cpu(),tree)
    for bm,bn in [(16,32),(16,64),(16,128),(32,64),(32,128),(64,64)]:
        def fn():
            _segment[(p.qids.shape[0],8,triton.cdiv(p.qids.shape[1]*4,bm))](
                qp,kp,vp,p.qids,p.kids,p.qlen,p.klen,p.part,p.lse,
                32,8,128,p.qids.shape[1],p.kids.shape[1],24,kp.shape[2],bm,bn,128,p.contiguous_k,p.contiguous_q,multibuffer=False)
        try:
            fn()
            _merge[(24,32)](p.part,p.lse,p.slots,p.counts,p.out,32,8,128,p.qids.shape[1],p.slots.shape[1],128)
            torch.testing.assert_close(p.out.cpu(),ref,atol=.008,rtol=.03)
            for _ in range(2):fn()
            start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(5):fn()
            end.record();end.synchronize()
            row={'case':name,'query_tile':bm,'key_tile':bn,'segment_us':start.elapsed_time(end)*1000/5}
            rows.append(row)
            print(json.dumps(row),flush=True)
            Path('results/tile_calibration.json').write_text(json.dumps(rows,indent=2))
        except Exception as exc:
            print(json.dumps({'case':name,'query_tile':bm,'key_tile':bn,'unsupported':type(exc).__name__}),flush=True)
