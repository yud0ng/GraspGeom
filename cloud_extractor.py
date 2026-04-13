"""
cloud_extractor.py
===================
从 SAM3 分割 mask + MoGe2 3D 点图中提取物体点云。

不再依赖 GrabCut：分割由 SAM3 负责，本模块只做：
  1. 将手部关节投影到 2D，生成手部排除 mask
  2. 用 (SAM3 mask & MoGe2 valid_mask & ~hand_mask) 索引点云
  3. 最小点数不足时在 mask bbox 内降级取点
"""

from __future__ import annotations

import cv2
import numpy as np


def project_points(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    将 3D 点投影到像素坐标。

    Args:
        points_3d: (N, 3) float，相机坐标系
        K:         (3, 3) 相机内参（像素空间）

    Returns:
        uv: (N, 2) int32
    """
    pts  = points_3d.T                   # (3, N)
    proj = K @ pts                       # (3, N)
    z    = np.clip(proj[2], 1e-6, None)
    uv   = (proj[:2] / z).T             # (N, 2)
    return uv.astype(np.int32)


def make_hand_mask(
    frame_shape: tuple,
    hand_uv: np.ndarray,
    dilation: int = 25,
) -> np.ndarray:
    """
    根据手部 2D 关键点生成手部排除 mask（凸包 + 膨胀）。

    Args:
        frame_shape: (H, W) 或 (H, W, C)
        hand_uv:     (N, 2) int，手部关节像素坐标
        dilation:    膨胀半径（像素）

    Returns:
        mask: (H, W) uint8，255 = 手部区域
    """
    H, W = frame_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    valid = (
        (hand_uv[:, 0] >= 0) & (hand_uv[:, 0] < W) &
        (hand_uv[:, 1] >= 0) & (hand_uv[:, 1] < H)
    )
    uv = hand_uv[valid]
    if len(uv) < 3:
        return mask

    hull = cv2.convexHull(uv.reshape(-1, 1, 2))
    cv2.fillPoly(mask, [hull], 255)

    if dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)

    return mask


def extract_object_cloud(
    moge_points: np.ndarray,       # (H, W, 3)
    moge_valid:  np.ndarray,       # (H, W) bool
    object_mask: np.ndarray,       # (H, W) uint8，SAM3 输出
    hand_mask:   np.ndarray | None = None,  # (H, W) uint8
    min_points:  int = 30,
) -> np.ndarray | None:
    """
    从 MoGe2 点图中按 mask 提取物体点云。

    Args:
        moge_points:  MoGe2 输出的 3D 点图 (H, W, 3)
        moge_valid:   MoGe2 有效深度 mask (H, W) bool
        object_mask:  SAM3 分割结果 (H, W) uint8（255=物体）
        hand_mask:    手部排除区域 (H, W) uint8；None 则不排除
        min_points:   返回点云的最小点数，不足则降级处理

    Returns:
        points_3d: (N, 3) float32；若点数仍不足则返回 None
    """
    select = (object_mask > 0) & moge_valid
    if hand_mask is not None:
        select &= ~(hand_mask > 0)

    points = moge_points[select]

    if len(points) >= min_points:
        return points.astype(np.float32)

    # ── 降级：使用 mask bbox 内全部有效点 ───────────
    ys, xs = np.where(object_mask > 0)
    if len(ys) == 0:
        return None

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    bbox_mask = np.zeros_like(object_mask)
    bbox_mask[y1:y2 + 1, x1:x2 + 1] = 255

    select_fb = (bbox_mask > 0) & moge_valid
    if hand_mask is not None:
        select_fb &= ~(hand_mask > 0)

    points_fb = moge_points[select_fb]
    if len(points_fb) < min_points:
        return None

    return points_fb.astype(np.float32)
