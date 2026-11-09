import json
import torch
import torch_npu
import tree_attention.qwen as integration
from tree_attention import KVTree
from tree_attention.reference import request_attention


def tensor_error(actual, reference):
    error=actual.float()-reference.float()
    return dict(max_abs_error=error.abs().max().item(),
                mean_abs_error=error.abs().mean().item(),
                relative_l2=(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(reference.float())).item())


def logit_error(actual,reference):
    result=tensor_error(actual,reference)
    result['top1_agreement']=(actual.argmax(-1)==reference.argmax(-1)).float().mean().item()
    return result


def context_paths(decoder):
    if isinstance(decoder,integration.CachedPrefixDecoder):
        return [path+tuple(decoder.prefix_size+r*decoder.max_new_tokens+t for t in range(decoder.generated))
                for r,path in enumerate(decoder.tree.paths)]
    return decoder.tree.paths


def vendor_attention(q,kp,vp,paths):
    outputs=[]
    for r,path in enumerate(paths):
        indices=torch.tensor(path,device=q.device)
        key=kp.index_select(2,indices).transpose(1,2).unsqueeze(0).contiguous()
        value=vp.index_select(1,indices).unsqueeze(0).contiguous()
        query=q[r].reshape(1,q.shape[1],1,q.shape[2]).contiguous()
        output=torch_npu.npu_fused_infer_attention_score(query,key,value,
            num_heads=q.shape[1],num_key_value_heads=kp.shape[0],input_layout='BNSD',scale=q.shape[2]**-.5)[0]
        outputs.append(output[0,:,0])
    return torch.stack(outputs)


def checked_step(decoder,tokens,vendor=False):
    original=integration.attention_matrix
    rows=[]
    def forward(q,kp,vp,prepared):
        paths=context_paths(decoder)
        vendor_result=vendor_attention(q,kp,vp,paths)
        if vendor:return vendor_result
        actual=original(q,kp,vp,prepared)
        reference=request_attention(q.cpu(),kp.permute(2,0,1).cpu(),vp.permute(1,0,2).cpu(),KVTree([],paths))
        host_actual=actual.cpu()
        row=dict(layer=len(rows),tree_vs_fp32=tensor_error(host_actual,reference),
                 vendor_vs_fp32=tensor_error(vendor_result.cpu(),reference),
                 tree_vs_vendor=tensor_error(host_actual,vendor_result.cpu()))
        rows.append(row)
        torch.testing.assert_close(host_actual,reference,atol=.008,rtol=.03,
                                   msg=f"Layer {row['layer']} attention: {row}")
        return actual
    integration.attention_matrix=forward
    try:
        logits=decoder.step(tokens).clone()
    finally:
        integration.attention_matrix=original
    return logits,rows


@torch.inference_mode()
def validate_pretrained(model,sequences,steps,planning):
    decoder_class=integration.CachedPrefixDecoder if planning=='cached' else integration.SharedKVDecoder
    rows=[];layers=[]
    for method in ('request','fixed','greedy','cube'):
        decoder=decoder_class(model,sequences,steps,method)
        vendor_decoder=decoder_class(model,sequences,steps,method)
        caches=[model(torch.tensor([s],device='npu'),use_cache=True).past_key_values for s in sequences]
        for step in range(steps):
            tokens=decoder.logits.argmax(-1).tolist()
            actual,layer_rows=checked_step(decoder,tokens)
            vendor,_=checked_step(vendor_decoder,tokens,vendor=True)
            full=[];standard=[]
            for r,sequence in enumerate(decoder.sequences):
                full.append(model(torch.tensor([sequence],device='npu'),use_cache=False).logits[0,-1])
                standard.append(model(torch.tensor([[tokens[r]]],device='npu'),past_key_values=caches[r],use_cache=True).logits[0,-1])
            full=torch.stack(full);standard=torch.stack(standard)
            row=dict(method=method,planning=planning,step=step,
                tree_vs_full=logit_error(actual,full),standard_cached_vs_full=logit_error(standard,full),
                tree_vs_standard_cached=logit_error(actual,standard),tree_vs_vendor=logit_error(actual,vendor),
                vendor_vs_full=logit_error(vendor,full),physical_kv_tokens=decoder.used,
                logical_kv_tokens=sum(map(len,decoder.sequences)))
            rows.append(row)
            for layer in layer_rows:
                layer.update(method=method,planning=planning,step=step)
                layers.append(layer)
            print(json.dumps(row),flush=True)
        del decoder,vendor_decoder,caches
    return dict(results=rows,layers=layers)
