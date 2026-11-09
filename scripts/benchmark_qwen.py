import argparse
import json
import time
from pathlib import Path
import torch
import torch_npu
from transformers import AutoModelForCausalLM,AutoTokenizer
from tree_attention.qwen import SharedKVDecoder,CachedPrefixDecoder


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--workloads',default='configs/qwen_workloads.json')
    parser.add_argument('--output',default='results/qwen_benchmark.json')
    parser.add_argument('--steps',type=int,default=16)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--planning',choices=['rebuild','cached'],default='cached')
    args=parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='eager').to('npu').eval()
    workloads=json.loads(Path(args.workloads).read_text())['workloads']
    rows=[]
    for name,sequences in workloads.items():
        baseline=None
        for method in ('request','fixed','greedy','cube'):
            decoder_class=CachedPrefixDecoder if args.planning=='cached' else SharedKVDecoder
            warmup=decoder_class(model,sequences,args.steps,method)
            for _ in range(2):
                warmup.step(warmup.logits.argmax(-1).tolist())
            torch.npu.synchronize()
            del warmup
            torch.npu.reset_peak_memory_stats()
            start=time.perf_counter()
            decoder=decoder_class(model,sequences,args.steps,method)
            torch.npu.synchronize()
            prefill_ms=(time.perf_counter()-start)*1000
            times=[]
            tokens=[decoder.logits.argmax(-1).tolist()]
            for step in range(args.steps - 1):
                torch.npu.synchronize()
                start=time.perf_counter()
                decoder.step(tokens[-1])
                selected=decoder.logits.argmax(-1).tolist()
                torch.npu.synchronize()
                times.append((time.perf_counter()-start)*1000)
                tokens.append(selected)
            if baseline is None:baseline=tokens
            agreement=sum(a==b for aa,bb in zip(tokens,baseline) for a,b in zip(aa,bb))/(len(sequences)*args.steps)
            row=dict(workload=name,method=method,planning=args.planning,seed=args.seed,generated_tokens=len(sequences)*args.steps,
                     new_tokens_per_request=args.steps,prefill_ms=prefill_ms,
                     decode_step_ms=times,decode_total_ms=sum(times),generation_total_ms=prefill_ms+sum(times),
                     decode_tokens_per_second=len(sequences)*(args.steps-1)/sum(times)*1000,
                     token_agreement_with_request=agreement,physical_kv_tokens=decoder.used,allocated_kv_tokens=decoder.capacity,
                     logical_kv_tokens=sum(map(len,decoder.sequences)),peak_allocated_bytes=torch.npu.max_memory_allocated(),
                     token_ids=[[step[r] for step in tokens] for r in range(len(sequences))],
                     continuations=[tokenizer.decode([step[r] for step in tokens]) for r in range(len(sequences))])
            rows.append(row)
            print(json.dumps(row),flush=True)
            Path(args.output).write_text(json.dumps({'model':args.model,'results':rows},indent=2))
            del decoder


if __name__=='__main__':main()
