from dataclasses import dataclass
import torch
import numpy as np
import triton
import triton.language as tl


@triton.jit
def _segment(Q, K, V, QIDS, KIDS, QLEN, KLEN, PART, LSE,
             HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
             MAXR: tl.constexpr, MAXK: tl.constexpr, NREQ: tl.constexpr, NTOK: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
             CONTIGUOUS_K: tl.constexpr, CONTIGUOUS_Q: tl.constexpr, JOB_BASE: tl.constexpr = 0):
    job = tl.program_id(0) + JOB_BASE
    head = tl.program_id(1)
    tile = tl.program_id(2)
    group: tl.constexpr = HQ // HK
    rows = tile * BM + tl.arange(0, BM)
    req_idx = rows // group
    nr = tl.load(QLEN + job)
    nk = tl.load(KLEN + job)
    dd = tl.arange(0, BD)
    if CONTIGUOUS_Q:
        first = tl.load(QIDS + job * MAXR)
        qq = tl.load(Q + ((head * NREQ * group + first * group + rows[:, None]) * D) + dd[None, :],
                     (rows[:, None] < nr * group) & (dd[None, :] < D), 0)
    else:
        rid = tl.load(QIDS + job * MAXR + req_idx, req_idx < nr, 0)
        qq = tl.load(Q + ((head * NREQ * group + rid[:, None] * group + rows[:, None] % group) * D) + dd[None, :],
                     (rows[:, None] < nr * group) & (dd[None, :] < D), 0)
    m = tl.full((BM,), -float('inf'), tl.float32)
    z = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    for start in range(0, tl.cdiv(nk, BN)):
        kk = start * BN + tl.arange(0, BN)
        if CONTIGUOUS_K:
            ids = tl.load(KIDS + job * MAXK) + kk
        else:
            ids = tl.load(KIDS + job * MAXK + kk, kk < nk, 0)
        kval = tl.load(K + (head * D + dd[:, None]) * NTOK + ids[None, :],
                       (kk[None, :] < nk) & (dd[:, None] < D), 0)
        scores = tl.dot(qq, kval) * (D ** -0.5)
        scores = tl.where(kk[None, :] < nk, scores, -float('inf'))
        new_m = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])
        vval = tl.load(V + (head * NTOK + ids[:, None]) * D + dd[None, :],
                       (kk[:, None] < nk) & (dd[None, :] < D), 0)
        acc = acc * alpha[:, None] + tl.dot(p.to(vval.dtype), vval)
        z = z * alpha + tl.sum(p, axis=1)
        m = new_m
    part_row = (job * HK + head) * (MAXR * group) + rows
    tl.store(PART + part_row[:, None] * D + dd[None, :],
             acc / z[:, None], (rows[:, None] < nr * group) & (dd[None, :] < D))
    tl.store(LSE + part_row, m + tl.log(z), rows < nr * group)


@triton.jit
def _merge(PART, LSE, SLOTS, COUNTS, OUT,
           HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
           MAXR: tl.constexpr, MAXS: tl.constexpr, BD: tl.constexpr):
    request = tl.program_id(0)
    head = tl.program_id(1)
    group: tl.constexpr = HQ // HK
    dd = tl.arange(0, BD)
    count = tl.load(COUNTS + request)
    m = tl.full((), -float('inf'), tl.float32)
    z = tl.full((), 0, tl.float32)
    acc = tl.zeros((BD,), tl.float32)
    for s in range(count):
        slot = tl.load(SLOTS + request * MAXS + s)
        job = slot // MAXR
        req = slot % MAXR
        row = (job * HK + head // group) * (MAXR * group) + req * group + head % group
        lse = tl.load(LSE + row)
        new_m = tl.maximum(m, lse)
        alpha = tl.exp(m - new_m)
        p = tl.exp(lse - new_m)
        part = tl.load(PART + row * D + dd, dd < D, 0)
        acc = acc * alpha + part * p
        z = z * alpha + p
        m = new_m
    tl.store(OUT + (request * HQ + head) * D + dd, acc / z, dd < D)


@dataclass
class PreparedPlan:
    jobs: list
    small_jobs: int
    contiguous_k: bool
    contiguous_q: bool
    qids: torch.Tensor
    kids: torch.Tensor
    qlen: torch.Tensor
    klen: torch.Tensor
    slots: torch.Tensor
    counts: torch.Tensor
    part: torch.Tensor
    lse: torch.Tensor
    out: torch.Tensor


def prepare(plan, q):
    """Materialize metadata and reusable scratch; include this cost when plans change."""
    from .schedule import physical_jobs
    jobs = physical_jobs(plan.jobs)
    kvheads = plan.settings["kvheads"]
    groups = q.shape[1] // kvheads
    jobs.sort(key=lambda job: len(job.requests) * groups > 16)
    small_jobs = sum(len(job.requests) * groups <= 16 for job in jobs)
    maxr = max(len(j.requests) for j in jobs)
    contiguous_k = all(j.tokens[-1] - j.tokens[0] + 1 == len(j.tokens) for j in jobs)
    maxk = 1 if contiguous_k else max(len(j.tokens) for j in jobs)
    qids = np.zeros((len(jobs), maxr), dtype=np.int32)
    kids = np.zeros((len(jobs), maxk), dtype=np.int32)
    slots = [[] for _ in range(q.shape[0])]
    for i, job in enumerate(jobs):
        qids[i, :len(job.requests)] = job.requests
        if contiguous_k:
            kids[i, 0] = job.tokens[0]
        else:
            kids[i, :len(job.tokens)] = job.tokens
        for j, r in enumerate(job.requests):
            slots[r].append(i * maxr + j)
    maxs = max(map(len, slots))
    slot_array = np.zeros((q.shape[0], maxs), dtype=np.int32)
    for r, values in enumerate(slots):
        slot_array[r, :len(values)] = values
    arrays = [qids, kids, np.array([len(j.requests) for j in jobs],dtype=np.int32),
              np.array([len(j.tokens) for j in jobs],dtype=np.int32), slot_array,
              np.array([len(s) for s in slots],dtype=np.int32)]
    dev = q.device
    metadata = torch.from_numpy(np.concatenate([a.ravel() for a in arrays])).to(dev)
    views, offset = [], 0
    for array in arrays:
        views.append(metadata[offset:offset+array.size].view(array.shape))
        offset += array.size
    contiguous_q = all(j.requests[-1] - j.requests[0] + 1 == len(j.requests) for j in jobs)
    return PreparedPlan(jobs, small_jobs, contiguous_k, contiguous_q, *views,
        torch.empty((len(jobs), maxr, q.shape[1], q.shape[2]), device=dev, dtype=torch.float32),
        torch.empty((len(jobs), maxr, q.shape[1]), device=dev, dtype=torch.float32), torch.empty_like(q))


def pack_kv(k, v):
    return k.permute(1,2,0).contiguous(), v.permute(1,0,2).contiguous()


def attention(q, k, v, prepared):
    return attention_matrix(q, *pack_kv(k,v), prepared)


def attention_matrix(q, kp, vp, prepared):
    """Attention on resident K[H,D,T] and V[H,T,D] physical layouts."""
    p = prepared
    hq, d = q.shape[1:]
    hk = kp.shape[0]
    qp = q.reshape(q.shape[0], hk, hq // hk, d).permute(1,0,2,3).contiguous()
    phases = [(0, p.small_jobs, 16, 128), (p.small_jobs, p.qids.shape[0] - p.small_jobs, 64, 64)]
    for base, count, bm, bn in phases:
        if count:
            qtiles = 1 if bm == 16 else triton.cdiv(p.qids.shape[1] * (hq // hk), bm)
            _segment[(count, hk, qtiles)](
                qp, kp, vp, p.qids, p.kids, p.qlen, p.klen, p.part, p.lse,
                hq, hk, d, p.qids.shape[1], p.kids.shape[1], q.shape[0], kp.shape[2],
                bm, bn, triton.next_power_of_2(d), p.contiguous_k, p.contiguous_q, base, multibuffer=False)
    _merge[(q.shape[0], hq)](p.part, p.lse, p.slots, p.counts, p.out,
        hq, hk, d, p.qids.shape[1], p.slots.shape[1], triton.next_power_of_2(d))
    return p.out


class CachedDecodePlan:
    """A prompt schedule plus one contiguous growing KV tail per request."""
    def __init__(self, prefix_plan, q, prefix_tokens, max_new_tokens):
        from .schedule import Job, Plan
        tails = [Job((r,),tuple(range(prefix_tokens+r*max_new_tokens,
                                     prefix_tokens+(r+1)*max_new_tokens)))
                 for r in range(q.shape[0])]
        plan = Plan(prefix_plan.jobs+tails, prefix_plan.method, prefix_plan.estimated_us, prefix_plan.settings)
        self.prepared = prepare(plan,q)
        ordered = self.prepared.jobs
        self.tail_indices = torch.tensor([i for i,job in enumerate(ordered)
                                         if job.tokens[0]>=prefix_tokens],device=q.device)

    def update(self, generated_tokens):
        self.prepared.klen.index_fill_(0,self.tail_indices,generated_tokens)
        return self.prepared
