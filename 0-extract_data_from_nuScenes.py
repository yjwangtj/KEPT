import json, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample", required=True)  # /data/nuscenes_dataset/v1.0-trainval/sample_data.json
parser.add_argument("--input", required=True)    # 0-sequential_scenes_train.json or 0-sequential_scenes_train.json
parser.add_argument("--output", required=True)     # output_path: 0-****_sample_val.json or 0-****_sample_train.json
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
