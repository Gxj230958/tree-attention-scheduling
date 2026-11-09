import argparse
import json
from dataclasses import asdict
from pathlib import Path
from tree_attention import KVTree, schedule

parser=argparse.ArgumentParser()
parser.add_argument('workloads')
parser.add_argument('--case',default='document_qa')
parser.add_argument('--method',default='cube')
parser.add_argument('--output',default='results/plan.json')
args=parser.parse_args()
data=json.loads(Path(args.workloads).read_text())
tree=KVTree.from_sequences(data['workloads'][args.case])
plan=schedule(tree,method=args.method)
Path(args.output).write_text(json.dumps(asdict(plan),indent=2))
