import argparse
import json
from pathlib import Path
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from tree_attention.qwen import CachedPrefixDecoder, SharedKVDecoder
from model_checks import checked_step, logit_error


def top_two(logits, tokenizer):
    values, indices = logits.float().topk(2, dim=-1)
    return [dict(token_ids=i, tokens=[tokenizer.decode([t]) for t in i], logits=v,
                 margin=v[0]-v[1]) for i,v in zip(indices.tolist(),values.tolist())]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',default='results/qwen4b_divergence.json')
    parser.add_argument('--workloads',default='configs/qwen_workloads.json')
    args=parser.parse_args()
    records=json.loads(Path(args.input).read_text())['results']
    base=next(r for r in records if r['workload']=='multi_branch' and r['method']=='request')
    alternate=next(r for r in records if r['workload']=='multi_branch' and r['method']=='fixed')
    differences=[(s,r) for r,(a,b) in enumerate(zip(base['token_ids'],alternate['token_ids']))
                 for s,(x,y) in enumerate(zip(a,b)) if x!=y]
    first=min(s for s,r in differences)
    requests=[r for s,r in differences if s==first]
    sequences=json.loads(Path(args.workloads).read_text())['workloads']['multi_branch']
    torch.npu.set_device(0)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='eager').to('npu').eval()
    decoder_class=CachedPrefixDecoder if base['planning']=='cached' else SharedKVDecoder
    results={}
    outputs={}
    for method in ('request','fixed'):
        decoder=decoder_class(model,sequences,base['new_tokens_per_request'],method)
        layers=[]
        for step in range(first):
            tokens=[seq[step] for seq in base['token_ids']]
            if step==first-1:
                logits,layers=checked_step(decoder,tokens)
            else:
                logits=decoder.step(tokens)
        outputs[method]=logits.clone()
        results[method]=dict(top_two=top_two(logits,tokenizer),layers=layers,
                             actual_tokens=logits.argmax(-1).tolist())
        del decoder
    histories=[s+base['token_ids'][r][:first] for r,s in enumerate(sequences)]
    full=torch.stack([model(torch.tensor([s],device='npu'),use_cache=False).logits[0,-1]
                      for s in histories])
    for method,output in outputs.items():
        results[method]['vs_full']=logit_error(output,full)
    result=dict(input=args.input,first_different_generated_index=first,requests=requests,
                histories=histories,full_top_two=top_two(full,tokenizer),methods=results,
                request_vs_fixed=logit_error(outputs['request'],outputs['fixed']))
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('histories','methods')}),flush=True)


if __name__=='__main__':
    main()
