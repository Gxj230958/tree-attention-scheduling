import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',default='configs/qwen_workloads.json')
    args=parser.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    paragraphs=json.loads(Path('data/squad-normans.json').read_text())['data'][0]['paragraphs']
    source='\n\n'.join(p['context'] for p in paragraphs[:4])
    records=[q for p in paragraphs[:4] for q in p['qas'] if not q['is_impossible']][:8]
    document=[tokenizer.encode('Read the following historical source and answer briefly.\n'+source+'\nQuestion: '+q['question']+'\nAnswer:') for q in records]
    dialogue=[]
    base='User: Read this source.\n'+source+'\nAssistant: I have read it.\n'
    for i,q in enumerate(records):
        turns=''.join('User: '+previous['question']+'\nAssistant: '+previous['answers'][0]['text']+'\n' for previous in records[:i//3])
        dialogue.append(tokenizer.encode(base+turns+'User: '+q['question']+'\nAssistant:'))
    branch_base=tokenizer.encode('Explain the evidence in the following source.\n'+paragraphs[3]['context']+'\nAnswer:')
    branches=[branch_base+tokenizer.encode(s) for s in ['The source','The history','First,','In this account,','Evidence suggests','The narrative','One explanation','The main point']]
    no_share=[]
    first_tokens=set()
    for paragraph in paragraphs[10:]:
        tokens=tokenizer.encode(paragraph['context'])
        if tokens[0] not in first_tokens:
            no_share.append(tokens)
            first_tokens.add(tokens[0])
        if len(no_share)==8:
            break
    data={'model':args.model,'source':'SQuAD 2.0 development, Normans article','workloads':{
        'document_qa':document,'multi_turn':dialogue,'multi_branch':branches,'no_share':no_share}}
    Path(args.output).write_text(json.dumps(data,indent=2))


if __name__=='__main__': main()
