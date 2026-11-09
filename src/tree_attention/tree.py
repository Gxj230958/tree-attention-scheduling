from dataclasses import dataclass, field


@dataclass
class Node:
    parent: int
    tokens: tuple[int, ...]
    children: list[int] = field(default_factory=list)
    terminal: list[int] = field(default_factory=list)
    requests: tuple[int, ...] = ()


@dataclass
class KVTree:
    nodes: list[Node]
    paths: list[tuple[int, ...]]
    token_ids: list[int] | None = None

    @classmethod
    def from_sequences(cls, sequences: list[list[int]]):
        """Build a compressed radix tree and a unique-token KV storage order."""
        if not sequences or any(not seq for seq in sequences):
            raise ValueError("Every decode request needs a nonempty context")
        nodes = [Node(-1, ())]
        token_ids = []
        paths = [None] * len(sequences)

        def add(parent, reqs, offset, prefix):
            ended = [r for r in reqs if len(sequences[r]) == offset]
            nodes[parent].terminal.extend(ended)
            for r in ended:
                paths[r] = prefix
            branches = {}
            for r in reqs:
                if len(sequences[r]) > offset:
                    branches.setdefault(sequences[r][offset], []).append(r)
            for branch in branches.values():
                end = offset + 1
                shortest = min(len(sequences[r]) for r in branch)
                while end < shortest and len({sequences[r][end] for r in branch}) == 1:
                    end += 1
                ids = tuple(range(len(token_ids), len(token_ids) + end - offset))
                token_ids.extend(sequences[branch[0]][offset:end])
                node_id = len(nodes)
                nodes.append(Node(parent, ids, requests=tuple(branch)))
                nodes[parent].children.append(node_id)
                add(node_id, branch, end, prefix + ids)
        add(0, list(range(len(sequences))), 0, ())
        nodes[0].requests = tuple(range(len(sequences)))
        return cls(nodes, paths, token_ids)

    @classmethod
    def synthetic(cls, parents, lengths, terminal_nodes):
        nodes, count = [], 0
        for i, (parent, length) in enumerate(zip(parents, lengths)):
            if (i == 0 and parent != -1) or (i > 0 and not 0 <= parent < i):
                raise ValueError("Parents must precede children")
            nodes.append(Node(parent, tuple(range(count, count + length))))
            count += length
            if parent >= 0:
                nodes[parent].children.append(i)
        paths = []
        for r, node in enumerate(terminal_nodes):
            nodes[node].terminal.append(r)
            path = ()
            while node >= 0:
                nodes[node].requests += (r,)
                path = nodes[node].tokens + path
                node = nodes[node].parent
            if not path:
                raise ValueError("Every decode request needs a nonempty context")
            paths.append(path)
        return cls(nodes, paths)

    def extend_leaves(self, leaves, token_indices):
        """Append one KV position per request, retaining compressed private paths."""
        for request,token in enumerate(token_indices):
            parent = leaves[request]
            node = self.nodes[parent]
            if node.requests == (request,) and not node.children:
                node.tokens += (token,)
            else:
                node.terminal.remove(request)
                child = len(self.nodes)
                self.nodes.append(Node(parent,(token,),terminal=[request],requests=(request,)))
                node.children.append(child)
                leaves[request] = child
            self.paths[request] += (token,)

    @property
    def storage_tokens(self):
        return sum(len(n.tokens) for n in self.nodes)
