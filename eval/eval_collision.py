import json
import math
import argparse
from typing import Dict, Tuple, Any, List

import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

def get_ego_pose_SE3(nusc: NuScenes, sample_token: str):

    sample = nusc.get('sample', sample_token)
    sd_token = sample['data'].get('LIDAR_TOP', None)
    if sd_token is None:
        sd_token = next(iter(sample['data'].values()))
    sd = nusc.get('sample_data', sd_token)
    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
    t = np.array(ego_pose['translation'], dtype=float)
    q = Quaternion(ego_pose['rotation'])
    R_e2g = q.rotation_matrix
    R_g2e = R_e2g.T
    return t, R_e2g, R_g2e

_pose_cache = {}
def get_ego_pose_SE3_cached(nusc: NuScenes, sample_token: str):
    if sample_token in _pose_cache:
        return _pose_cache[sample_token]
    val = get_ego_pose_SE3(nusc, sample_token)
    _pose_cache[sample_token] = val
    return val

def chain_next_samples(nusc: NuScenes, sample_token: str, steps: int = 6) -> List[str]:
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
        try:
            pred_like = json.loads(pred_like)
        except Exception:
            raise ValueError(f"cannot load: {pred_like}")

    out = {}
    targets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    if isinstance(pred_like, list):
        if len(pred_like) == 0:
            return out
        if all(isinstance(v, (list, tuple)) for v in pred_like):
            for i, v in enumerate(pred_like[:6]):
                out[i+1] = (float(v[0]), float(v[1]))
            return out
        if all(isinstance(v, dict) for v in pred_like):
            pred_like = pred_like[0]

        if all(isinstance(v, str) for v in pred_like):
            pred_like = json.loads(pred_like[0])

    if isinstance(pred_like, dict):
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

            idx = min(range(6), key=lambda i: abs(targets[i] - t)) + 1
            out[idx] = (x, y)
        return out

    raise TypeError(f"error format: {type(pred_like)}")


def extract_pred_field(rec: Dict[str, Any]) -> Dict[int, Tuple[float, float]]:
    for key in ['predict_traj']:
        if key in rec and rec[key] not in (None, ''):
            return normalize_pred_dict(rec[key])
    raise KeyError("no match keys")


def yaw_from_R(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))

def yaw_from_quat(q: Quaternion) -> float:
    return yaw_from_R(q.rotation_matrix)

def wrap_pi(a: float) -> float:
    a = (a + np.pi) % (2 * np.pi) - np.pi
    return float(a)

def rect_corners_xy(center: Tuple[float, float], length: float, width: float, yaw: float) -> np.ndarray:
    cx, cy = center
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[ hl,  hw],
                      [ hl, -hw],
                      [-hl, -hw],
                      [-hl,  hw]], dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s,  c]], dtype=float)
    return (local @ R.T) + np.array([cx, cy], dtype=float)

def _project_on_axis(pts: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    scal = pts @ axis
    return float(np.min(scal)), float(np.max(scal))

def obb_overlap_2d(c1: np.ndarray, c2: np.ndarray) -> bool:
    axes = []
    e1 = c1[1] - c1[0]
    e2 = c1[3] - c1[0]
    axes.append(e1); axes.append(e2)
    f1 = c2[1] - c2[0]
    f2 = c2[3] - c2[0]
    axes.append(f1); axes.append(f2)
    for ax in axes:
        min1, max1 = _project_on_axis(c1, ax)
        min2, max2 = _project_on_axis(c2, ax)
        if max1 < min2 or max2 < min1:
            return False
    return True

def z_overlap(zc1: float, h1: float, zc2: float, h2: float) -> bool:
    half1, half2 = h1 / 2.0, h2 / 2.0
    return (zc1 - half1) <= (zc2 + half2) and (zc2 - half2) <= (zc1 + half1)

def get_ann_boxes_in_t0(nusc: NuScenes,
                        t0_token: str,
                        tok_at_step: str,
                        allowed_prefix: Tuple[str, ...],
                        R_g2e_0: np.ndarray,
                        t0_global: np.ndarray,
                        inflate: float = 0.0):
    sample = nusc.get('sample', tok_at_step)
    out = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        cat = ann.get('category_name', '')
        if allowed_prefix and not any(cat.startswith(p) for p in allowed_prefix):
            continue

        ctr_global = np.array(ann['translation'], dtype=float)
        d_global = ctr_global - t0_global
        d_body0 = R_g2e_0.dot(d_global)
        x, y, z = float(d_body0[0]), float(d_body0[1]), float(d_body0[2])

        w, l, h = [float(s) for s in ann['size']] 
        if inflate != 0.0:
            w = max(0.0, w + inflate * 2.0)
            l = max(0.0, l + inflate * 2.0)

        q_box = Quaternion(ann['rotation'])
        yaw_box_global = yaw_from_quat(q_box)
        _, R_e2g_0, _ = get_ego_pose_SE3_cached(nusc, t0_token)
        yaw0_global = yaw_from_R(R_e2g_0)
        yaw_rel = wrap_pi(yaw_box_global - yaw0_global)

        out.append({
            "center_xy": (x, y),
            "yaw": yaw_rel,
            "width": w,
            "length": l,
            "z": z,
            "height": h
        })
    return out

def evaluate_counts_for_record(nusc: NuScenes,
                               rec: Dict[str, Any],
                               ego_length: float,
                               ego_width: float,
                               ego_height: float,
                               inflate: float,
                               allowed_prefix: Tuple[str, ...],
                               ignore_z: bool) -> Dict[str, Any]:
    st = rec.get('sample_token') or rec.get('token')
    preds = extract_pred_field(rec)
    next_tokens = chain_next_samples(nusc, st, steps=6)

    t0, R_e2g_0, R_g2e_0 = get_ego_pose_SE3_cached(nusc, st)

    yaw_pred = {}
    for i in range(1, 7):
        if i in preds and (i - 1) in preds:
            x1, y1 = preds[i - 1]; x2, y2 = preds[i]
            yaw_pred[i] = math.atan2(y2 - y1, x2 - x1)
        elif i in preds:
            x2, y2 = preds[i]
            yaw_pred[i] = math.atan2(y2, x2)
        else:
            yaw_pred[i] = None

    per_step_collisions = []
    per_step_possible   = []

    for i, tok in enumerate(next_tokens, start=1):
        p = preds.get(i, None)
        if p is None:
            per_step_collisions.append(None)
            per_step_possible.append(None)
            continue

        ann_boxes = get_ann_boxes_in_t0(
            nusc=nusc,
            t0_token=st,
            tok_at_step=tok,
            allowed_prefix=allowed_prefix,
            R_g2e_0=R_g2e_0,
            t0_global=t0,
            inflate=inflate
        )

        obs_cnt = len(ann_boxes)
        if obs_cnt == 0:
            per_step_collisions.append(0)
            per_step_possible.append(0)
            continue

        yaw_p = yaw_pred.get(i) if yaw_pred.get(i) is not None else 0.0
        ego_xy = (float(p[0]), float(p[1]))
        corners_pred = rect_corners_xy(
            ego_xy,
            length=float(ego_length + 2 * inflate),
            width=float(ego_width + 2 * inflate),
            yaw=yaw_p
        )
        zc_pred, h_pred = 0.0, float(max(0.0, ego_height))

        colliders = 0
        for box in ann_boxes:
            corners_obs = rect_corners_xy(
                center=box["center_xy"],
                length=box["length"],
                width=box["width"],
                yaw=box["yaw"]
            )
            if not obb_overlap_2d(corners_pred, corners_obs):
                continue
            if ignore_z:
                colliders += 1
            else:
                if z_overlap(zc_pred, h_pred, box["z"], box["height"]):
                    colliders += 1

        per_step_collisions.append(int(colliders))
        per_step_possible.append(int(obs_cnt))

    for _ in range(len(per_step_collisions), 6):
        per_step_collisions.append(None)
        per_step_possible.append(None)

    return {
        "sample_token": st,
        "per_step_collisions": per_step_collisions,
        "per_step_possible": per_step_possible
    }

def eval_file(dataroot: str,
              version: str,
              input_json: str,
              ego_length: float = 4.084,
              ego_width: float = 1.850,
              ego_height: float = 1.50,
              inflate: float = 0.0,
              allowed_prefix_str: str = "vehicle,human,animal,movable_object",
              ignore_z: bool = False):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    with open(input_json, 'r', encoding='utf-8') as f:
        records = json.load(f)

    allowed_prefix = tuple(s.strip() for s in allowed_prefix_str.split(",") if s.strip())

    step_collisions_total = [0] * 6
    step_possible_total   = [0] * 6

    for rec in records:
        out = evaluate_counts_for_record(
            nusc=nusc,
            rec=rec,
            ego_length=ego_length,
            ego_width=ego_width,
            ego_height=ego_height,
            inflate=inflate,
            allowed_prefix=allowed_prefix,
            ignore_z=ignore_z
        )
        for i in range(6):
            c = out["per_step_collisions"][i]
            p = out["per_step_possible"][i]
            if c is None or p is None:
                continue
            step_collisions_total[i] += int(c)
            step_possible_total[i]   += int(p)

    cum_collisions = []
    cum_possible   = []
    run_c = 0
    run_p = 0
    for i in range(6):
        run_c += step_collisions_total[i]
        run_p += step_possible_total[i]
        cum_collisions.append(run_c)
        cum_possible.append(run_p)

    rates = []
    for i in range(6):
        if cum_possible[i] > 0:
            rates.append(cum_collisions[i] / cum_possible[i])
        else:
            rates.append(float('nan'))

    print(' '.join(f'{r:.6f}' if math.isfinite(r) else '0.0000' for r in rates))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/data/nuscenes_dataset", help="nuScenes root path")
    ap.add_argument("--version", default="v1.0-trainval", choices=["v1.0-trainval","v1.0-mini","v1.0-test"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--ego-length", type=float, default=4.084)
    ap.add_argument("--ego-width", type=float, default=1.850)
    ap.add_argument("--ego-height", type=float, default=1.50)
    ap.add_argument("--inflate", type=float, default=0.1)
    ap.add_argument("--allowed-prefix", type=str, default="vehicle,human,animal,movable_object")
    ap.add_argument("--ignore-z", action="store_true")
    args = ap.parse_args()

    eval_file(
        dataroot=args.dataroot,
        version=args.version,
        input_json=args.input,
        ego_length=args.ego_length,
        ego_width=args.ego_width,
        ego_height=args.ego_height,
        inflate=args.inflate,
        allowed_prefix_str=args.allowed_prefix,
        ignore_z=args.ignore_z
    )

