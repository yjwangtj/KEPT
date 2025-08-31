import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--basedata", required=True)    # 1-****_val.json
parser.add_argument("--pred", required=True)        # 4-**output**.json
parser.add_argument("--output", required=True)      # 5-**eval**.json
args = parser.parse_args()

with open(args.basedata, 'r') as f1:
    ls1 = json.load(f1)
with open(args.pred, 'r') as f2:
    ls2 = json.load(f2)

datas = []
for obj1, obj2 in zip(ls1, ls2):
    if obj2.get('response') is not None:
        datas.append({**obj1, "predict_traj": obj2['response']})
    else:
        pass

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(datas, f, indent=4, ensure_ascii=False)
