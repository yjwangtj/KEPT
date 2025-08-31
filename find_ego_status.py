from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from scipy.spatial.transform import Rotation as R
import numpy as np

DATAROOT = "/data/nuscenes_dataset"
VERSION = "v1.0-trainval"

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
nusc_can = NuScenesCanBus(dataroot=DATAROOT)

def _nearest_by_utime(msgs, target_utime):

    if not msgs:
        return None, None
    idx = np.argmin([abs(m["utime"] - target_utime) for m in msgs])
    m = msgs[idx]
    return m, int(m["utime"] - target_utime)

def can_from_sample_token(sample_token):
    try:
        sample = nusc.get("sample", sample_token)
        scene = nusc.get("scene", sample["scene_token"])
        scene_name = scene["name"]
        t_us = sample["timestamp"]
    except Exception:

        return {
            "scene_name": None,
            "speed_mps": None,
            "accel_mps2_xyz": None,
            "yaw_rad": None,
            "time_offset_us": None
        }

    try:

        vm_msgs   = nusc_can.get_messages(scene_name, "vehicle_monitor")
        pose_msgs = nusc_can.get_messages(scene_name, "pose")
        imu_msgs  = nusc_can.get_messages(scene_name, "ms_imu")
    except Exception:

        return {
            "scene_name": scene_name,
            "speed_mps": None,
            "accel_mps2_xyz": None,
            "yaw_rad": None,
            "time_offset_us": None
        }


    speed_ms, accel_xyz, yaw_rad, dt_align = None, None, None, None

    vm_near, dt_vm = _nearest_by_utime(vm_msgs, t_us)
    if vm_near is not None and "vehicle_speed" in vm_near:
        try:
            speed_ms = float(vm_near["vehicle_speed"])
        except Exception:
            pass

    pose_near, dt_pose = _nearest_by_utime(pose_msgs, t_us)
    if pose_near is not None:
        if "accel" in pose_near and pose_near["accel"] is not None:
            try:
                accel_xyz = np.array(pose_near["accel"], dtype=float).tolist()
            except Exception:
                accel_xyz = None
        if "orientation" in pose_near and pose_near["orientation"] is not None:
            ori = pose_near["orientation"]
            try:
                if len(ori) == 4:
                    try:
                        yaw_rad = R.from_quat([ori[1], ori[2], ori[3], ori[0]]).as_euler("zyx")[0]
                    except Exception:
                        yaw_rad = R.from_quat([ori[0], ori[1], ori[2], ori[3]]).as_euler("zyx")[0]
                elif len(ori) == 3:
                    yaw_rad = float(ori[2])
            except Exception:
                yaw_rad = None

    if accel_xyz is None and imu_msgs:
        imu_near, _ = _nearest_by_utime(imu_msgs, t_us)
        if imu_near is not None:
            for k in ("linear_accel", "accel"):
                if k in imu_near and imu_near[k] is not None:
                    try:
                        accel_xyz = np.array(imu_near[k], dtype=float).tolist()
                    except Exception:
                        accel_xyz = None
                    break

    dt_align = dt_pose if pose_near is not None else (dt_vm if vm_near is not None else None)

    return {
        "scene_name": scene_name,
        "speed_mps": speed_ms,
        "accel_mps2_xyz": accel_xyz,
        "yaw_rad": yaw_rad,
        "time_offset_us": dt_align
    }

