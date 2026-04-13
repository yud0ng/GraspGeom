"""
pose_estimator.py
==================
对物体点云做 PCA，估计三个主轴方向、尺寸及有向包围盒（OBB）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class ObjectAxes:
    """物体主轴分析结果（相机坐标系）。"""

    center:    np.ndarray   # (3,) 质心
    n_points:  int          # 参与 PCA 的有效点数

    # 三个主轴（单位向量，按方差降序排列）
    long_axis:  np.ndarray  # (3,) 第 1 主成分（最长方向）
    mid_axis:   np.ndarray  # (3,) 第 2 主成分
    short_axis: np.ndarray  # (3,) 第 3 主成分（最短方向）

    # 对应方向的尺寸估计（2σ ≈ 覆盖 95% 范围，单位与深度图一致）
    long_size:  float
    mid_size:   float
    short_size: float

    explained_variance:       np.ndarray  # (3,) 特征值
    explained_variance_ratio: np.ndarray  # (3,) 方差占比

    obb_vertices: np.ndarray  # (8, 3) OBB 顶点，相机坐标系

    def axis_matrix(self) -> np.ndarray:
        """返回 (3, 3)，列为 [long, mid, short] 方向向量。"""
        return np.stack([self.long_axis, self.mid_axis, self.short_axis], axis=1)

    def to_dict(self) -> dict:
        return {
            'center':                    self.center.tolist(),
            'n_points':                  self.n_points,
            'long_axis':                 self.long_axis.tolist(),
            'mid_axis':                  self.mid_axis.tolist(),
            'short_axis':                self.short_axis.tolist(),
            'long_size':                 float(self.long_size),
            'mid_size':                  float(self.mid_size),
            'short_size':                float(self.short_size),
            'explained_variance':        self.explained_variance.tolist(),
            'explained_variance_ratio':  self.explained_variance_ratio.tolist(),
            'obb_vertices':              self.obb_vertices.tolist(),
        }


def estimate_object_axes(
    points: np.ndarray,
    n_std: float = 2.0,
    outlier_std: float = 3.0,
) -> ObjectAxes | None:
    """
    对物体点云做 PCA，返回主轴方向和 OBB。

    Args:
        points:      (N, 3) float，相机坐标系物体点云
        n_std:       尺寸估计倍数（n_std × σ × 2，2.0 ≈ 覆盖 95%）
        outlier_std: 离群点阈值（到质心距离超过此倍数 σ 的点被剔除）

    Returns:
        ObjectAxes，或点数不足时返回 None
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 10:
        return None

    # ── 去离群点 ────────────────────────────────────
    if len(pts) > 30:
        center_tmp = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center_tmp, axis=1)
        thresh = dists.mean() + outlier_std * dists.std()
        pts = pts[dists < thresh]
        if len(pts) < 10:
            return None

    # ── PCA ─────────────────────────────────────────
    pca = PCA(n_components=3)
    pca.fit(pts)

    center = pca.mean_.astype(np.float32)
    axes   = pca.components_.astype(np.float32)   # (3, 3) 行=主轴，降序
    ev     = pca.explained_variance_.astype(np.float32)
    evr    = pca.explained_variance_ratio_.astype(np.float32)

    long_axis, mid_axis, short_axis = axes[0], axes[1], axes[2]

    # ── 沿各轴的投影标准差 → 尺寸 ─────────────────
    pts_c = pts - center
    long_size  = float(n_std * 2 * (pts_c @ long_axis).std())
    mid_size   = float(n_std * 2 * (pts_c @ mid_axis).std())
    short_size = float(n_std * 2 * (pts_c @ short_axis).std())

    # ── OBB 8 顶点 ───────────────────────────────────
    half = np.array([long_size / 2, mid_size / 2, short_size / 2])
    signs = np.array([
        [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
        [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
    ], dtype=np.float32)
    corners_local = signs * half
    R = np.stack([long_axis, mid_axis, short_axis], axis=0)  # (3, 3) 行=轴
    obb_vertices = (corners_local @ R + center).astype(np.float32)

    return ObjectAxes(
        center=center,
        n_points=len(pts),
        long_axis=long_axis,
        mid_axis=mid_axis,
        short_axis=short_axis,
        long_size=long_size,
        mid_size=mid_size,
        short_size=short_size,
        explained_variance=ev,
        explained_variance_ratio=evr,
        obb_vertices=obb_vertices,
    )
