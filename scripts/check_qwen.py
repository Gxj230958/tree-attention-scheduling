import argparse
import json
from pathlib import Path
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from tree_attention.qwen import SharedKVDecoder,CachedPrefixDecoder


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='tiny')
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--planning',choices=['rebuild','cached'],default='rebuild')
    parser.add_argument('--output',default='results/qwen_correctness.json')
    parser.add_argument('--workloads')
    parser.add_argument('--workload',default='document_qa')
    args=parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(2026)
    if args.model == 'tiny':
        config=Qwen3Config(vocab_size=128,hidden_size=128,intermediate_size=256,num_hidden_layers=2,
                          num_attention_heads=4,num_key_value_heads=2,head_dim=32)
        model=Qwen3ForCausalLM(config).to(dtype=torch.float16,device='npu').eval()
        sequences=[[1,2,3,4,5],[1,2,3,8],[1,2,3,4,5],[7,8,9,10,11,12]]
        tokenizer=None
    else:
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
                        torch_dtype=torch.bfloat16,attn_implementation='eager').to('npu').eval()
        prefixes=['You are a helpful assistant. A researcher is studying attention kernels. ',
                  'You are a helpful assistant. A researcher is studying attention kernels. ',
                  'The museum opens at nine and closes at five. ', 'Give a short definition of entropy. ']
        endings=['Explain shared prefix caching.','Explain numerical stability.','When does it close?','']
        sequences=[tokenizer.encode(p+e) for p,e in zip(prefixes,endings)]
        if args.workloads:
            sequences=json.loads(Path(args.workloads).read_text())['workloads'][args.workload]
        from model_checks import validate_pretrained
        result=validate_pretrained(model,sequences,args.steps,args.planning)
        result.update(model=args.model,workload=args.workload if args.workloads else 'short_contexts')
        Path(args.output).write_text(json.dumps(result,indent=2))
        return
    rows=[]
    for method in ('request','fixed','greedy','cube'):
        decoder_class=CachedPrefixDecoder if args.planning=='cached' else SharedKVDecoder
        decoder=decoder_class(model,sequences,args.steps,method)
        token_history=[]
        for step in range(args.steps):
            tokens=decoder.logits.argmax(-1).tolist()
            actual=decoder.step(tokens)
            references=[]
            with torch.inference_mode():
                for sequence in decoder.sequences:
                    references.append(model(torch.tensor([sequence],device='npu'),use_cache=False).logits[0,-1])
            reference=torch.stack(references)
            error=(actual.float()-reference.float()).abs()
            torch.testing.assert_close(actual,reference,atol=.004,rtol=.02)
            rows.append(dict(method=method,planning=args.planning,step=step,max_abs_error=error.max().item(),
                             mean_abs_error=error.mean().item(),argmax_match=(actual.argmax(-1)==reference.argmax(-1)).float().mean().item(),
                             physical_kv_tokens=decoder.used,logical_kv_tokens=sum(map(len,decoder.sequences))))
            token_history.append(tokens)
            print(json.dumps(rows[-1]),flush=True)
        if tokenizer:
            print(json.dumps({'method':method,'continuations':[tokenizer.decode([step[r] for step in token_history]) for r in range(len(sequences))]}),flush=True)
        del decoder
    Path(args.output).write_text(json.dumps({'model':args.model,'results':rows},indent=2))


if __name__=='__main__':
    main()
