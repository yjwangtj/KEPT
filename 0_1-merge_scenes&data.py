import json, argparse
from find_ego_status import can_from_sample_token

parser = argparse.ArgumentParser()
parser.add_argument("--sample", required=True)   #  0-****_sample_val.json          or 0-****_sample_train.json
parser.add_argument("--scenes",   required=True)   #  0-sequential_scenes_val.json    or 0-sequential_scenes_train.json
parser.add_argument("--out",    required=True)   #  1-aligned_scenes_data_val.json  or 1-aligned_scenes_data_train.json
args = parser.parse_args()

with open(args.sample, "r", encoding="utf-8") as f1:
    ls1 = json.load(f1)
with open(args.scenes, "r", encoding="utf-8") as f2:
    ls2 = json.load(f2)

datas = []
for obj1, obj2 in zip(ls1, ls2):
    status = can_from_sample_token(obj1["sample_token"])
    if not status or any(v is None for v in status.values()):
        continue

    datas.append(
        {
            "token": obj1["token"],
            "sample_token": obj1["sample_token"],
            "ego_pose_token": obj1["ego_pose_token"],
            "calibrated_sensor_token": obj1["calibrated_sensor_token"],
            "prev": obj1["prev"],
            "next": obj1["next"],
            "images": obj2["images"],
            "new_trajectories": obj2["new_trajectories"],
            "history_trajectories": obj2["history_trajectories"],
            "navi": obj2["navi"],
            "status": status,
        }
    )

with open(args.out, "w", encoding="utf-8") as f:
    json.dump(datas, f, indent=4, ensure_ascii=False)

print(f"[Info] saved {len(datas)} samples。")
