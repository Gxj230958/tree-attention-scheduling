from dataclasses import dataclass, asdict
from functools import lru_cache
from math import ceil
from .tree import KVTree


@dataclass(frozen=True)
class Job:
    requests: tuple[int, ...]
    tokens: tuple[int, ...]


def physical_jobs(jobs):
    """Partition logical segments into contiguous physical KV reads."""
    result = []
    for job in jobs:
        if job.tokens[-1] - job.tokens[0] + 1 == len(job.tokens):
            result.append(job)
            continue
        start = 0
        for i in range(1,len(job.tokens)):
            if job.tokens[i] != job.tokens[i-1] + 1:
                result.append(Job(job.requests,job.tokens[start:i]))
                start = i
        result.append(Job(job.requests,job.tokens[start:]))
    return result


@dataclass
class CostModel:
    """Analytical roofline parameters in bytes/us and FLOP/us."""
    bandwidth: float = 900_000.0
    cube_flops: float = 100_000_000.0
    task_us: float = 0.15
    merge_us: float = 0.03
    cube_rows: int = 16
    key_tile: int = 128
    kv_split: int = 512
    max_query_rows: int = 64

    def cost(self, requests, tokens, qheads, kvheads, dim):
        if not requests or not tokens:
            return 0.0
        rows = len(requests) * (qheads // kvheads)
        query_tile = self.cube_rows if rows <= self.cube_rows else self.max_query_rows
        key_tile = self.key_tile if rows <= self.cube_rows else self.key_tile // 2
        qtiles = ceil(rows / query_tile)
        ktiles = ceil(len(tokens) / key_tile)
        flops = 4 * qtiles * query_tile * ktiles * key_tile * dim * kvheads
        reads = 4 * len(tokens) * dim * kvheads * qtiles
        partial = 4 * len(requests) * qheads * (dim + 1)
        return max(flops / self.cube_flops, reads / self.bandwidth) + self.task_us * qtiles * kvheads + partial / self.bandwidth + self.merge_us * len(requests)


@dataclass
class Plan:
    jobs: list[Job]
    method: str
    estimated_us: float
    settings: dict

    def validate(self, tree):
        covered = [[] for _ in tree.paths]
        for job in self.jobs:
            for request in job.requests:
                covered[request].extend(job.tokens)
        for request, path in enumerate(tree.paths):
            if sorted(covered[request]) != sorted(path):
                raise ValueError(f"Plan does not partition request {request}'s context")


def schedule(tree: KVTree, qheads=32, kvheads=8, dim=128, method="cube", model=None):
    if qheads % kvheads:
        raise ValueError("Query heads must be divisible by KV heads")
    model = model or CostModel()
    groups = qheads // kvheads
    req_tile = max(1, model.max_query_rows // groups)

    def split(requests, tokens):
        return [Job(tuple(requests[q:q+req_tile]), tuple(tokens[k:k+model.kv_split]))
                for q in range(0, len(requests), req_tile)
                for k in range(0, len(tokens), model.kv_split)]

    def cost(jobs):
        return sum(model.cost(j.requests, j.tokens, qheads, kvheads, dim) for j in physical_jobs(jobs))

    if method == "request":
        jobs = [j for r, path in enumerate(tree.paths) for j in split((r,), path)]
    elif method == "fixed":
        jobs = [j for n in tree.nodes for j in split(n.requests, n.tokens)]
    elif method in ("greedy", "cube"):
        @lru_cache(None)
        def count_cost(count, tokens):
            return cost(split(range(count),tokens))

        @lru_cache(None)
        def solve(node_id, inherited):
            node = tree.nodes[node_id]
            tokens = inherited + node.tokens
            if not node.children:
                return count_cost(len(node.requests),tokens), ()
            if method == "greedy":
                remaining = len(node.requests)
                pushed = []
                for child in node.children:
                    descendant = tree.nodes[child]
                    count = len(descendant.requests)
                    split_k = count_cost(remaining,tokens) + count_cost(count,descendant.tokens)
                    split_q = count_cost(remaining-count,tokens) + count_cost(count,tokens+descendant.tokens)
                    if split_q < split_k:
                        pushed.append(child)
                        remaining -= count
                total = count_cost(remaining,tokens)
                total += sum(solve(child,tokens if child in pushed else ())[0] for child in node.children)
                return total,tuple(pushed)
            states = {0:(0.0,())}
            for child in node.children:
                count = len(tree.nodes[child].requests)
                shared_cost = solve(child,())[0]
                pushed_cost = solve(child,tokens)[0]
                updated = {}
                for removed,(subtotal,pushed) in states.items():
                    for extra,subcost,selected in ((0,shared_cost,pushed),
                                                   (count,pushed_cost,pushed+(child,))):
                        key = removed+extra
                        value = subtotal+subcost
                        if key not in updated or value < updated[key][0]:
                            updated[key] = value,selected
                states = updated
            return min((subtotal+count_cost(len(node.requests)-removed,tokens),pushed)
                       for removed,(subtotal,pushed) in states.items())

        jobs = []
        def materialize(node_id,inherited):
            node = tree.nodes[node_id]
            tokens = inherited+node.tokens
            _,pushed = solve(node_id,inherited)
            detached = {r for child in pushed for r in tree.nodes[child].requests}
            remaining = tuple(r for r in node.requests if r not in detached)
            jobs.extend(split(remaining,tokens))
            for child in node.children:
                materialize(child,tokens if child in pushed else ())
        materialize(0,())
        # Release recursive planning state at the end of this invocation.
        del solve, materialize
    else:
        raise ValueError(f"Unknown scheduling method: {method}")
    settings = asdict(model)
    settings.update(qheads=qheads, kvheads=kvheads, head_dim=dim)
    result = Plan(jobs, method, cost(jobs), settings)
    return result
