"""
grasp_analyzer.py
==================
分析手部与物体的相对几何关系，估计接触点候选。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pose_estimator import ObjectAxes


@dataclass
class GraspGeometry:
    """手部-物体相对几何关系。"""

    wrist_pos:      np.ndarray   # (3,) 腕部位置（相机坐标系）
    approach_dir:   np.ndarray   # (3,) 手掌接近方向（归一化）
    wrist_to_obj:   np.ndarray   # (3,) 腕 → 物体质心的向量
    dist_wrist_obj: float        # 腕到质心距离

    # 接近方向与各主轴夹角（度，取绝对值 [0°, 90°]）
    angle_vs_long:  float
    angle_vs_mid:   float
    angle_vs_short: float

    # 接近方向在物体主轴坐标系中的分量
    approach_in_obj_frame: np.ndarray   # (3,)

    # 腕部在物体主轴坐标系中的坐标（相对于质心）
    wrist_in_obj_frame: np.ndarray      # (3,)

    # 主导接近轴（0=长轴, 1=中轴, 2=短轴）
    dominant_axis_idx:   int
    dominant_axis_comp:  float

    def to_dict(self) -> dict:
        return {
            'wrist_pos':             self.wrist_pos.tolist(),
            'approach_dir':          self.approach_dir.tolist(),
            'wrist_to_obj':          self.wrist_to_obj.tolist(),
            'dist_wrist_obj':        self.dist_wrist_obj,
            'angle_vs_long':         self.angle_vs_long,
            'angle_vs_mid':          self.angle_vs_mid,
            'angle_vs_short':        self.angle_vs_short,
            'approach_in_obj_frame': self.approach_in_obj_frame.tolist(),
            'wrist_in_obj_frame':    self.wrist_in_obj_frame.tolist(),
            'dominant_axis_idx':     self.dominant_axis_idx,
            'dominant_axis_comp':    self.dominant_axis_comp,
        }

    def describe(self) -> str:
        names = ['长轴', '中轴', '短轴']
        dom   = names[self.dominant_axis_idx]
        side  = '正' if self.wrist_in_obj_frame[self.dominant_axis_idx] > 0 else '负'
        return (
            f"腕部: ({self.wrist_pos[0]:.3f}, {self.wrist_pos[1]:.3f}, {self.wrist_pos[2]:.3f})\n"
            f"腕→物体距离: {self.dist_wrist_obj:.3f} m\n"
            f"接近方向 vs 长/中/短轴: "
            f"{self.angle_vs_long:.1f}° / {self.angle_vs_mid:.1f}° / {self.angle_vs_short:.1f}°\n"
            f"主导接近轴: {dom}（{side}向）"
        )


def _abs_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """两向量夹角（度），取绝对值 [0°, 90°]。"""
    n1 = np.linalg.norm(v1) + 1e-9
    n2 = np.linalg.norm(v2) + 1e-9
    cos_a = float(np.clip(abs(float(v1 @ v2)) / (n1 * n2), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def analyze_grasp_geometry(
    obj_axes:     ObjectAxes,
    wrist_pos:    np.ndarray,
    approach_dir: np.ndarray,
) -> GraspGeometry:
    """
    计算手部相对于物体的抓取几何关系。

    Args:
        obj_axes:     物体主轴分析结果
        wrist_pos:    (3,) 腕部位置
        approach_dir: (3,) 手掌接近方向（掌心朝向，已归一化）

    Returns:
        GraspGeometry
    """
    d = np.linalg.norm(approach_dir) + 1e-9
    approach_dir = approach_dir / d

    wrist_to_obj = obj_axes.center - wrist_pos
    dist = float(np.linalg.norm(wrist_to_obj))

    ang_long  = _abs_angle(approach_dir, obj_axes.long_axis)
    ang_mid   = _abs_angle(approach_dir, obj_axes.mid_axis)
    ang_short = _abs_angle(approach_dir, obj_axes.short_axis)

    R_obj = np.stack([obj_axes.long_axis,
                      obj_axes.mid_axis,
                      obj_axes.short_axis], axis=0)   # (3, 3)

    approach_local = (R_obj @ approach_dir).astype(np.float32)
    wrist_local    = (R_obj @ (wrist_pos - obj_axes.center)).astype(np.float32)

    dom_idx  = int(np.argmax(np.abs(approach_local)))
    dom_comp = float(approach_local[dom_idx])

    return GraspGeometry(
        wrist_pos=wrist_pos.astype(np.float32),
        approach_dir=approach_dir.astype(np.float32),
        wrist_to_obj=wrist_to_obj.astype(np.float32),
        dist_wrist_obj=dist,
        angle_vs_long=ang_long,
        angle_vs_mid=ang_mid,
        angle_vs_short=ang_short,
        approach_in_obj_frame=approach_local,
        wrist_in_obj_frame=wrist_local,
        dominant_axis_idx=dom_idx,
        dominant_axis_comp=dom_comp,
    )


def estimate_contact_points(
    obj_axes:     ObjectAxes,
    approach_dir: np.ndarray,
    n_candidates: int = 4,
) -> np.ndarray:
    """
    沿接近方向在 OBB 6 个面中心选取候选接触点。

    Args:
        obj_axes:     物体主轴
        approach_dir: (3,) 接近方向
        n_candidates: 返回候选数量

    Returns:
        contact_pts: (n_candidates, 3)
    """
    approach_dir = approach_dir / (np.linalg.norm(approach_dir) + 1e-9)

    half = np.array([obj_axes.long_size / 2,
                     obj_axes.mid_size  / 2,
                     obj_axes.short_size / 2])

    face_normals = np.array([
         obj_axes.long_axis, -obj_axes.long_axis,
         obj_axes.mid_axis,  -obj_axes.mid_axis,
         obj_axes.short_axis, -obj_axes.short_axis,
    ])
    face_offsets = np.array([half[0], half[0],
                             half[1], half[1],
                             half[2], half[2]])
    face_centers = obj_axes.center + face_normals * face_offsets[:, None]

    # 与接近方向对齐度最高（夹角最小）的面优先
    alignment = face_normals @ approach_dir   # (6,)
    order     = np.argsort(alignment)         # 最负在前（最接近反向 = 最受挤压的面）

    return face_centers[order[:n_candidates]]
