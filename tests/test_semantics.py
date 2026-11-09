import unittest
import gc
import weakref
import torch
from tree_attention import KVTree, CostModel, schedule
from tree_attention.schedule import physical_jobs,Plan
from tree_attention.reference import request_attention, segmented_attention


class TreeAttentionTests(unittest.TestCase):
    def test_unequal_nested_and_terminal_requests(self):
        tree = KVTree.from_sequences([[1,2,3], [1,2], [1,2,4,5,6], [7,8], [1,2,3]])
        self.assertEqual(tree.storage_tokens, 8)
        self.compare(tree)

    def test_no_shared_prefix(self):
        self.compare(KVTree.from_sequences([[1,2,3], [4], [5,6,7,8,9]]))

    def test_deep_tree_and_extreme_logits(self):
        tree = KVTree.synthetic([-1,0,1,1,0], [129,65,17,33,81], [2,3,4])
        self.compare(tree, scale=30)

    def compare(self, tree, scale=1):
        torch.manual_seed(17)
        q = torch.randn(len(tree.paths), 8, 32) * scale
        k = torch.randn(tree.storage_tokens, 2, 32)
        v = torch.randn_like(k)
        ref = request_attention(q,k,v,tree)
        for method in ("request", "fixed", "greedy", "cube"):
            plan = schedule(tree, 8, 2, 32, method, CostModel(kv_split=48))
            plan.validate(tree)
            plan = Plan(physical_jobs(plan.jobs),method,plan.estimated_us,plan.settings)
            plan.validate(tree)
            torch.testing.assert_close(segmented_attention(q,k,v,plan), ref, atol=2e-5, rtol=2e-5)

    def test_extension_keeps_private_paths_compressed(self):
        tree = KVTree.from_sequences([[1,2],[1,2],[3,4]])
        leaves = [next(i for i,n in enumerate(tree.nodes) if r in n.terminal) for r in range(3)]
        tree.extend_leaves(leaves,[4,5,6])
        self.assertEqual(len(tree.nodes),5)
        tree.extend_leaves(leaves,[7,8,9])
        self.assertEqual(len(tree.nodes),5)
        self.compare(tree)

    def test_dynamic_plan_minimizes_global_estimated_cost(self):
        for root in (16,128,1024):
            for child in (8,64,256):
                for leaf in (16,128,512):
                    tree = KVTree.synthetic([-1,0,0,1,1,2,2],
                        [root,child,child,leaf,2*leaf,leaf,2*leaf],[3,4,5,6])
                    greedy = schedule(tree,method="greedy")
                    dynamic = schedule(tree,method="cube")
                    dynamic.validate(tree)
                    self.assertLessEqual(dynamic.estimated_us,greedy.estimated_us+1e-10)

    def test_schedule_releases_input_tree(self):
        gc.disable()
        try:
            for method in ("request","fixed","greedy","cube"):
                tree = KVTree.from_sequences([[1,2,3],[1,2,4]])
                reference = weakref.ref(tree)
                plan = schedule(tree,method=method)
                del tree
                self.assertIsNone(reference(),method)
                self.assertTrue(plan.jobs)
        finally:
            gc.enable()

    def test_plan_detects_double_counting(self):
        tree = KVTree.from_sequences([[1,2],[1,3]])
        plan = schedule(tree)
        plan.jobs.append(plan.jobs[0])
        with self.assertRaises(ValueError):
            plan.validate(tree)


if __name__ == "__main__":
    unittest.main()
