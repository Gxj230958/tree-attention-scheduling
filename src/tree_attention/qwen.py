from types import MethodType
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from .tree import KVTree
from .schedule import schedule, CostModel
from .kernels import prepare, attention_matrix, CachedDecodePlan


class SharedKVDecoder:
    """Qwen3 decode with a physical shared-prefix KV pool per layer."""
    def __init__(self, model, sequences, max_new_tokens=32, method="cube", cost_model=None):
        self.model = model
        self.sequences = [list(s) for s in sequences]
        self.tree = KVTree.from_sequences(self.sequences)
        self.method = method
        self.cost_model = cost_model or CostModel()
        self.device = next(model.parameters()).device
        config = model.config
        self.hq, self.hk = config.num_attention_heads, config.num_key_value_heads
        self.dim = config.head_dim
        self.original = [layer.self_attn.forward for layer in model.model.layers]
        self.max_new_tokens = max_new_tokens
        self.capacity = self.tree.storage_tokens + len(sequences) * max_new_tokens
        self.used = self.tree.storage_tokens
        self.prefix_size = self.used
        self.generated = 0
        self.pools = [(torch.empty((self.hk, self.dim, self.capacity), device=self.device, dtype=model.dtype),
                       torch.empty((self.hk, self.capacity, self.dim), device=self.device, dtype=model.dtype))
                      for _ in model.model.layers]
        self.leaves = [next(i for i,n in enumerate(self.tree.nodes) if r in n.terminal)
                       for r in range(len(sequences))]
        self._prefill()

    @torch.inference_mode()
    def _prefill(self):
        assigned = set()
        logits = []
        for request, seq in enumerate(self.sequences):
            output = self.model(torch.tensor([seq], device=self.device), use_cache=True)
            logits.append(output.logits[0,-1].clone())
            path = self.tree.paths[request]
            positions = [i for i,token in enumerate(path) if token not in assigned]
            destinations = [path[i] for i in positions]
            if positions:
                source = torch.tensor(positions, device=self.device)
                target = torch.tensor(destinations, device=self.device)
                for i,layer in enumerate(output.past_key_values.layers):
                    self.pools[i][0].index_copy_(2, target, layer.keys[0].index_select(1,source).transpose(1,2))
                    self.pools[i][1].index_copy_(1, target, layer.values[0].index_select(1,source))
            assigned.update(path)
        self.logits = torch.stack(logits)

    def _next_positions(self, count):
        positions = [self.prefix_size+r*self.max_new_tokens+self.generated for r in range(count)]
        self.generated += 1
        return positions

    def _prepare_plan(self):
        plan = schedule(self.tree,self.hq,self.hk,self.dim,self.method,self.cost_model)
        qshape = torch.empty((len(self.sequences),self.hq,self.dim), device=self.device, dtype=self.model.dtype)
        return prepare(plan,qshape)

    def _append(self, token_ids):
        start = self.used
        if start + len(token_ids) > self.capacity:
            raise ValueError("Decode exceeds allocated max_new_tokens")
        positions = self._next_positions(len(token_ids))
        for sequence,token in zip(self.sequences,token_ids):
            sequence.append(token)
        self.tree.extend_leaves(self.leaves,positions)
        self.used += len(token_ids)
        return positions

    @torch.inference_mode()
    def step(self, token_ids):
        token_ids = list(token_ids)
        positions = torch.tensor([[len(s)] for s in self.sequences], device=self.device)
        targets = torch.tensor(self._append(token_ids), device=self.device)
        prepared = self._prepare_plan()
        pools = self.pools

        def forward(module, hidden_states, position_embeddings, attention_mask=None, **kwargs):
            shape = (*hidden_states.shape[:-1],-1,module.head_dim)
            q = module.q_norm(module.q_proj(hidden_states).view(shape)).transpose(1,2)
            k = module.k_norm(module.k_proj(hidden_states).view(shape)).transpose(1,2)
            v = module.v_proj(hidden_states).view(shape).transpose(1,2)
            q,k = apply_rotary_pos_emb(q,k,*position_embeddings)
            kp,vp = pools[module.layer_idx]
            kp.index_copy_(2,targets,k[:,:,0,:].permute(1,2,0))
            vp.index_copy_(1,targets,v[:,:,0,:].permute(1,0,2))
            result = attention_matrix(q[:,:,0,:].contiguous(),kp,vp,prepared)
            return module.o_proj(result.reshape(len(token_ids),1,-1)),None

        for layer in self.model.model.layers:
            layer.self_attn.forward = MethodType(forward,layer.self_attn)
        try:
            output = self.model(torch.tensor(token_ids,device=self.device)[:,None],
                                position_ids=positions,use_cache=False)
            self.logits = output.logits[:,0]
        finally:
            for layer,original in zip(self.model.model.layers,self.original):
                layer.self_attn.forward = original
        return self.logits


class CachedPrefixDecoder(SharedKVDecoder):
    """Reuse prompt groups and update only the length of each private decode tail."""
    def __init__(self, model, sequences, max_new_tokens=32, method="cube", cost_model=None):
        super().__init__(model,sequences,max_new_tokens,method,cost_model)
        prefix = schedule(self.tree,self.hq,self.hk,self.dim,self.method,self.cost_model)
        qshape = torch.empty((len(sequences),self.hq,self.dim),device=self.device,dtype=model.dtype)
        self.cached_plan = CachedDecodePlan(prefix,qshape,self.prefix_size,max_new_tokens)

    def _append(self, token_ids):
        if self.generated >= self.max_new_tokens:
            raise ValueError("Decode exceeds allocated max_new_tokens")
        positions = self._next_positions(len(token_ids))
        for sequence,token in zip(self.sequences,token_ids):
            sequence.append(token)
        self.used += len(token_ids)
        return positions

    def _prepare_plan(self):
        return self.cached_plan.update(self.generated)
