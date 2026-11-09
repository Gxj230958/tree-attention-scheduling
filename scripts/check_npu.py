import json
import torch
import torch_npu
from tree_attention import KVTree, CostModel, schedule
from tree_attention.kernels import prepare, attention
from tree_attention.reference import request_attention


def main():
    torch.npu.set_device(0)
    torch.manual_seed(2026)
    trees = {
        'unequal': KVTree.synthetic([-1,0,0,1,1], [129,65,17,33,81], [2,3,4]),
        'none': KVTree.from_sequences([[1,2,3], [4], [5,6,7,8,9]]),
        'duplicate_terminal': KVTree.from_sequences([[1,2], [1,2], [1,2,3,4], [1,5,6]]),
        'wide': KVTree.synthetic([-1]+[0]*32, [257]+[31+i for i in range(32)], list(range(1,33))),
    }
    rows = []
    for name, tree in trees.items():
        for dtype in (torch.float16, torch.bfloat16):
            q = torch.randn(len(tree.paths), 8, 64, dtype=dtype)
            k = torch.randn(tree.storage_tokens, 2, 64, dtype=dtype)
            v = torch.randn_like(k)
            ref = request_attention(q,k,v,tree)
            q,k,v = (t.npu() for t in (q,k,v))
            for method in ('request','fixed','greedy','cube'):
                plan = schedule(tree,8,2,64,method,CostModel(kv_split=128))
                result = attention(q,k,v,prepare(plan,q)).cpu()
                error = (result.float()-ref.float()).abs().max().item()
                torch.testing.assert_close(result, ref, atol=.006, rtol=.02)
                rows.append(dict(case=name,dtype=str(dtype),method=method,max_abs_error=error,jobs=len(plan.jobs)))
                print(json.dumps(rows[-1]), flush=True)
    with open('results/npu_correctness.json','w') as f:
        json.dump(rows,f,indent=2)


if __name__ == '__main__':
    main()
