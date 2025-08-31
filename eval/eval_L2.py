import json
import argparse
import math
import numpy as np
from typing import Dict, Tuple, Any
from nuscenes import NuScenes
from pyquaternion import Quaternion

'''
Usage:
    python eval_L2.py --input 5-**eval**.json 
'''

def get_ego_pose_SE3(nusc: NuScenes, sample_token: str):

    sample = nusc.get('sample', sample_token)
    sd_token = sample['data'].get('LIDAR_TOP', None)
    if sd_token is None:
        sd_token = next(iter(sample['data'].values()))
    sd = nusc.get('sample_data', sd_token)
    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])

    t = np.array(ego_pose['translation'], dtype=float)
    q = Quaternion(ego_pose['rotation'])
    R_ego2global = q.rotation_matrix
    R_global2ego = R_ego2global.T
    return t, R_ego2global, R_global2ego


def chain_next_samples(nusc: NuScenes, sample_token: str, steps: int = 6):
    out = []
    cur = nusc.get('sample', sample_token)
    for _ in range(steps):
        nxt = cur['next']
        if not nxt:
            break
        cur = nusc.get('sample', nxt)
        out.append(cur['token'])
    return out

def normalize_pred_dict(pred_like: Any) -> Dict[int, Tuple[float, float]]:
    if isinstance(pred_like, str):
        pred_like = json.loads(pred_like)

    out = {}
    targets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for k, v in pred_like.items():
        if isinstance(v, dict):
            x, y = float(v['x']), float(v['y'])
        else:
            x, y = float(v[0]), float(v[1])
        s = str(k).lower().replace('seconds', '').replace('second', '').replace('s', '').strip()
        try:
            t = float(s.split()[0])
        except Exception:
            t = float(s)
        idx = min(range(6), key=lambda i: abs(targets[i] - t)) + 1  # 1..6
        out[idx] = (x, y)
    return out


def project_future_gt_to_t0(nusc: NuScenes, sample_token: str, steps: int = 6) -> Dict[int, Tuple[float, float]]:
    t0, R_e2g_0, R_g2e_0 = get_ego_pose_SE3(nusc, sample_token)
    next_tokens = chain_next_samples(nusc, sample_token, steps=steps)
    gt = {}
    for i, tok in enumerate(next_tokens, start=1):
        t_i, _, _ = get_ego_pose_SE3(nusc, tok)
        d_global = t_i - t0
        d_body_0 = R_g2e_0.dot(d_global)
        gt[i] = (float(d_body_0[0]), float(d_body_0[1]))  # 只用平面
    return gt


def l2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))


def extract_pred_field(rec: Dict[str, Any]) -> Dict[int, Tuple[float, float]]:
    for key in ['predict_traj']:
        if key in rec and rec[key] not in (None, ''):
            return normalize_pred_dict(rec[key])
    raise KeyError("no match found")


def eval_file(dataroot: str, version: str, input_json: str, dump_json: str = None):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    with open(input_json, 'r', encoding='utf-8') as f:
        records = json.load(f)

    results = []
    print(f"Loaded {len(records)} records from: {input_json}")
    for idx, rec in enumerate(records, 1):
        st = rec.get('sample_token') or rec.get('token')
        if not st:
            print(f"[{idx}] skip sample_token/token")
            continue

        try:
            preds = extract_pred_field(rec)        # {1..6:(x,y)}
            gts   = project_future_gt_to_t0(nusc, st, steps=6)  # {1..6:(x,y)}
        except Exception as e:
            print(f"[{idx}] failed {st}: {e}")
            continue

        row = {"sample_token": st, "per_step": {}, "ADE": None, "FDE": None, "valid_steps": 0}
        l2_list = []

        for i in range(1, 7):
            p, g = preds.get(i), gts.get(i)
            if p is None or g is None:
                row["per_step"][f"{0.5*i:.1f}s"] = None
            else:
                e = l2(p, g)
                row["per_step"][f"{0.5*i:.1f}s"] = e
                l2_list.append(e)

        if l2_list:
            row["ADE"] = float(np.mean(l2_list))
            row["FDE"] = float(l2_list[-1])
            row["valid_steps"] = len(l2_list)

        results.append(row)

        def _fmt4(x):
            return "None" if x is None else f"{x:.4f}"

        per_step_str = " | ".join(f"{k}:{_fmt4(v)}" for k, v in row["per_step"].items())
        ade_str = _fmt4(row["ADE"])
        fde_str = _fmt4(row["FDE"])
        print(f"[{idx}] {st} -> {per_step_str} || ADE:{ade_str} FDE:{fde_str}")

    if dump_json:
        with open(dump_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nsaved：{dump_json}")

    all_steps = ["0.5s","1.0s","1.5s","2.0s","2.5s","3.0s"]
    sums = {k: 0.0 for k in all_steps}
    cnts = {k: 0   for k in all_steps}
    for r in results:
        for k, v in r["per_step"].items():
            if v is not None:
                sums[k] += v
                cnts[k] += 1
    means = []
    for k in all_steps:  # ["0.5s","1.0s","1.5s","2.0s","2.5s","3.0s"]
        mean = (sums[k] / cnts[k]) if cnts[k] > 0 else float('nan')
        means.append(mean)

    print(' '.join(f'{m:.4f}' if math.isfinite(m) else '0.0000' for m in means))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default= "/data/nuscenes_dataset", help="nuScenes root path")
    ap.add_argument("--version", default="v1.0-trainval", choices=["v1.0-trainval","v1.0-mini","v1.0-test"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    eval_file(args.dataroot, args.version, args.input, dump_json=args.output)
