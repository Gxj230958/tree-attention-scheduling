from itertools import product
from tree_attention import KVTree


def evaluation_trees():
    """Depth-first storage and held-out lengths, spanning Cube row boundaries."""
    for fanout, root, shared, private in product(((3,1),(9,3),(17,7)), (32,192,768),
                                                (12,96,384), (24,192,768)):
        parents, lengths, terminals = [-1], [root], []
        for branch, width in enumerate(fanout):
            node = len(parents)
            parents.append(0)
            lengths.append(shared * (branch+1) + 7*branch)
            for leaf in range(width):
                terminals.append(len(parents))
                parents.append(node)
                lengths.append(private * (1+leaf%3) + 5*branch)
        parameters = dict(fanout=fanout, root=root, shared=shared, private=private)
        yield parameters, KVTree.synthetic(parents,lengths,terminals)
