import torch


def request_attention(q, k, v, tree):
    outputs = []
    groups = q.shape[1] // k.shape[1]
    for r, path in enumerate(tree.paths):
        ids = torch.tensor(path, device=k.device)
        kr = k.index_select(0, ids).repeat_interleave(groups, dim=1)
        vr = v.index_select(0, ids).repeat_interleave(groups, dim=1)
        scores = torch.einsum("hd,khd->hk", q[r].float(), kr.float()) / q.shape[-1] ** 0.5
        outputs.append(torch.einsum("hk,khd->hd", scores.softmax(-1), vr.float()))
    return torch.stack(outputs).to(q.dtype)


def segmented_attention(q, k, v, plan):
    """Portable FP32 oracle with global log-sum-exp normalization."""
    groups = q.shape[1] // k.shape[1]
    partial = [[] for _ in range(q.shape[0])]
    for job in plan.jobs:
        qs = q[list(job.requests)].float()
        ks = k[list(job.tokens)].repeat_interleave(groups, 1).float()
        vs = v[list(job.tokens)].repeat_interleave(groups, 1).float()
        scores = torch.einsum("rhd,khd->rhk", qs, ks) / q.shape[-1] ** 0.5
        lse = torch.logsumexp(scores, -1)
        out = torch.einsum("rhk,khd->rhd", scores.softmax(-1), vs)
        for i, request in enumerate(job.requests):
            partial[request].append((out[i], lse[i]))
    result = []
    for parts in partial:
        out, lse = map(torch.stack, zip(*parts))
        result.append((out * lse.softmax(0)[..., None]).sum(0))
    return torch.stack(result).to(q.dtype)
