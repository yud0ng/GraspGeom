"""
hand_state.py
==============
从 VITRA hand_state.npy 中提取手部关节位置和接近方向。

hand_state 格式（来自 VITRA 重建）:
    data['right'][frame_id] = {
        'transl':        np.ndarray (3,)        腕部位置（相机坐标系，单位米）
        'global_orient': np.ndarray (3, 3)      全局旋转矩阵
        'hand_pose':     np.ndarray (15, 3, 3)  15 关节旋转矩阵
        'beta':          np.ndarray (10,)       形状参数
    }

MANO 关节顺序（共 21 点，0=腕，1-20=手指）:
    0  : Wrist
    1  : Index MCP      5  : Middle MCP    9  : Ring MCP     13 : Pinky MCP
    2  : Index PIP      6  : Middle PIP   10  : Ring PIP     14 : Pinky PIP
    3  : Index DIP      7  : Middle DIP   11  : Ring DIP     15 : Pinky DIP
    4  : Index TIP      8  : Middle TIP   12  : Ring TIP     16 : Pinky TIP
    17 : Thumb CMC     18  : Thumb MCP    19  : Thumb IP     20 : Thumb TIP
"""

from __future__ import annotations
from pathlib import Path
import numpy as np


# MANO 骨架拓扑：每个关节的父关节索引（-1 = 根节点/腕部）
MANO_PARENT = [
    -1,           # 0  Wrist
     0,  1,  2,  3,   # 1-4   Index
     0,  5,  6,  7,   # 5-8   Middle
     0,  9, 10, 11,   # 9-12  Ring
     0, 13, 14, 15,   # 13-16 Pinky
     0, 17, 18, 19,   # 17-20 Thumb
]

# MANO 默认骨骼长度（右手，单位米，MANO 论文平均尺寸）
MANO_DEFAULT_BONE_LEN = np.array([
    0.0,
    0.095, 0.038, 0.030, 0.023,   # Index
    0.090, 0.042, 0.028, 0.022,   # Middle
    0.085, 0.040, 0.026, 0.022,   # Ring
    0.080, 0.032, 0.024, 0.020,   # Pinky
    0.050, 0.038, 0.028, 0.022,   # Thumb
], dtype=np.float32)


def load_hand_state(path: str | Path) -> dict:
    """加载 hand_state.npy，返回原始 dict。"""
    return np.load(str(path), allow_pickle=True).item()


def get_frame_data(hand_state: dict, frame_id: int,
                   hand: str = 'right') -> dict | None:
    """获取某帧的手部重建数据，不存在则返回 None。"""
    return hand_state.get(hand, {}).get(frame_id)


def get_available_frames(hand_state: dict, hand: str = 'right') -> list[int]:
    """返回有重建结果的帧 ID 列表（升序）。"""
    return sorted(hand_state.get(hand, {}).keys())


def get_camera_params(hand_state: dict) -> dict:
    """提取相机参数：width, height, fov_x, fps。"""
    return {
        'width':  int(hand_state['width']),
        'height': int(hand_state['height']),
        'fov_x':  float(hand_state.get('fov_x', 0.0)),
        'fps':    float(hand_state.get('fps', 30.0)),
    }


def get_wrist_and_approach(frame_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    从帧数据中提取腕部位置和手掌接近方向。

    接近方向 = global_orient 的 -Z 列（MANO 约定：Z 轴从掌心向外，
    故 -Z 为掌心朝向，即手接近物体的方向）。

    Returns:
        wrist_pos:    (3,) float32，腕部在相机坐标系的位置（米）
        approach_dir: (3,) float32，归一化接近方向
    """
    wrist_pos     = np.array(frame_data['transl'],        dtype=np.float32)
    global_orient = np.array(frame_data['global_orient'], dtype=np.float32)

    palm_normal = -global_orient[:, 2]
    norm = np.linalg.norm(palm_normal)
    if norm > 1e-9:
        palm_normal /= norm

    return wrist_pos, palm_normal


def get_hand_joints(frame_data: dict) -> np.ndarray:
    """
    通过简化前向运动学计算 21 个 MANO 关节的 3D 位置。

    Returns:
        joints: (21, 3) float32，相机坐标系（单位米）
    """
    transl        = np.array(frame_data['transl'],        dtype=np.float32)
    global_orient = np.array(frame_data['global_orient'], dtype=np.float32)
    hand_pose     = np.array(frame_data['hand_pose'],     dtype=np.float32)

    # 构建完整旋转矩阵：腕部(global_orient) + 15 关节 + 5 个 TIP（单位矩阵）
    all_rot = [global_orient] + [hand_pose[i] for i in range(15)] \
              + [np.eye(3, dtype=np.float32)] * 5

    joints      = np.zeros((21, 3), dtype=np.float32)
    global_rots = [np.eye(3, dtype=np.float32)] * 21
    joints[0]   = transl

    bone_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    for j in range(1, 21):
        p = MANO_PARENT[j]
        rot = all_rot[j] if j < len(all_rot) else np.eye(3, dtype=np.float32)
        global_rots[j] = global_rots[p] @ rot
        d = global_rots[p] @ bone_dir
        joints[j] = joints[p] + MANO_DEFAULT_BONE_LEN[j] * d

    return joints
