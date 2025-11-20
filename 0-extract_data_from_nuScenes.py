import json, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample", required=True)
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.sample, "r", encoding="utf-8") as f:
    datas = json.load(f)
with open(args.input, "r", encoding="utf-8") as f:
    vals = json.load(f)

index = {d["filename"]: d for d in datas}

res = []
for val in vals:
    key = val["images"][6]
    new_key = key.split("/", 3)[3]
    if new_key in index:
        res.append(index[new_key])

with open(args.out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=4, ensure_ascii=False)

