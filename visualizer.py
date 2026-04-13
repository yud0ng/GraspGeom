"""
visualizer.py
==============
将物体主轴、OBB、手部接近方向、接触点等叠加到视频帧上，
并提供可选的 Open3D 3D 可视化。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pose_estimator import ObjectAxes
    from grasp_analyzer import GraspGeometry


# ─────────────────────────────────────────────────────────
# 2D 叠加绘制
# ─────────────────────────────────────────────────────────

def _project(pt3d: np.ndarray, K: np.ndarray) -> tuple[int, int] | None:
    p = K @ pt3d.astype(np.float64)
    if p[2] < 1e-6:
        return None
    return (int(p[0] / p[2]), int(p[1] / p[2]))


def draw_object_mask(
    frame: np.ndarray,
    mask:  np.ndarray,
    color: tuple = (0, 200, 100),
    alpha: float = 0.35,
) -> np.ndarray:
    """半透明叠加物体 mask（BGR 颜色）。"""
    vis     = frame.copy()
    overlay = frame.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    return vis


def draw_axes(
    frame: np.ndarray,
    obj_axes,
    K:     np.ndarray,
    scale: float = 0.05,
    thickness: int = 2,
) -> np.ndarray:
    """绘制物体三主轴（长=红, 中=绿, 短=蓝）。"""
    vis    = frame.copy()
    c3d    = obj_axes.center
    c2d    = _project(c3d, K)
    if c2d is None:
        return vis

    cv2.circle(vis, c2d, 5, (0, 255, 255), -1)

    for axis, size, color, label in [
        (obj_axes.long_axis,  obj_axes.long_size,  (0,   0,   255), 'L'),
        (obj_axes.mid_axis,   obj_axes.mid_size,   (0,   255, 0),   'M'),
        (obj_axes.short_axis, obj_axes.short_size, (255, 0,   0),   'S'),
    ]:
        half     = min(size / 2, scale * max(c3d[2], 0.1))
        pt_pos   = _project(c3d + axis * half, K)
        pt_neg   = _project(c3d - axis * half, K)
        if pt_pos and pt_neg:
            cv2.arrowedLine(vis, c2d, pt_pos, color, thickness, tipLength=0.3)
            cv2.line(vis, c2d, pt_neg, color, max(1, thickness // 2))
            cv2.putText(vis, label, pt_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis


def draw_obb(
    frame:       np.ndarray,
    obb_vertices: np.ndarray,   # (8, 3)
    K:           np.ndarray,
    color:       tuple = (0, 200, 200),
    thickness:   int   = 1,
) -> np.ndarray:
    """绘制有向包围盒线框。"""
    vis   = frame.copy()
    edges = [
        (0,1),(2,3),(4,5),(6,7),
        (0,2),(1,3),(4,6),(5,7),
        (0,4),(1,5),(2,6),(3,7),
    ]
    pts2d = [_project(obb_vertices[i], K) for i in range(8)]
    for i, j in edges:
        if pts2d[i] and pts2d[j]:
            cv2.line(vis, pts2d[i], pts2d[j], color, thickness)
    return vis


def draw_approach(
    frame:    np.ndarray,
    grasp_geo,
    K:        np.ndarray,
    length:   float = 0.08,
    color:    tuple = (255, 128, 0),
    thickness: int  = 2,
) -> np.ndarray:
    """绘制手掌接近方向箭头（橙色）。"""
    vis   = frame.copy()
    wrist = grasp_geo.wrist_pos
    tip   = wrist + grasp_geo.approach_dir * length
    p0    = _project(wrist, K)
    p1    = _project(tip,   K)
    if p0 and p1:
        cv2.arrowedLine(vis, p0, p1, color, thickness, tipLength=0.25)
        cv2.putText(vis, 'approach', p1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def draw_contact_points(
    frame:       np.ndarray,
    contact_pts: np.ndarray,   # (N, 3)
    K:           np.ndarray,
) -> np.ndarray:
    """绘制候选接触点（青色系）。"""
    vis    = frame.copy()
    colors = [(0,255,255), (0,200,200), (0,150,150), (0,100,100)]
    for i, pt in enumerate(contact_pts):
        p2d = _project(pt, K)
        if p2d:
            c = colors[i % len(colors)]
            cv2.circle(vis, p2d, 7, c, -1)
            cv2.putText(vis, f'C{i}', (p2d[0] + 8, p2d[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return vis


def annotate_frame(
    frame:       np.ndarray,
    obj_axes,
    grasp_geo,
    contact_pts: np.ndarray,
    obj_mask:    np.ndarray,
    K:           np.ndarray,
    frame_id:    int,
    object_name: str = "",
) -> np.ndarray:
    """综合叠加所有信息，返回可视化帧。"""
    vis = draw_object_mask(frame, obj_mask)
    vis = draw_obb(vis, obj_axes.obb_vertices, K)
    vis = draw_axes(vis, obj_axes, K)
    if grasp_geo is not None:
        vis = draw_approach(vis, grasp_geo, K)
        vis = draw_contact_points(vis, contact_pts, K)

    lines = [
        f"Frame {frame_id}" + (f"  [{object_name}]" if object_name else ""),
        f"Center: ({obj_axes.center[0]:.2f}, {obj_axes.center[1]:.2f}, {obj_axes.center[2]:.2f})",
        f"L/M/S: {obj_axes.long_size:.3f} / {obj_axes.mid_size:.3f} / {obj_axes.short_size:.3f} m",
    ]
    if grasp_geo is not None:
        lines += [
            f"Approach vs Long: {grasp_geo.angle_vs_long:.1f} deg",
            f"Dist wrist-obj:   {grasp_geo.dist_wrist_obj:.3f} m",
        ]

    for i, line in enumerate(lines):
        y = 25 + i * 22
        cv2.putText(vis, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,   0,   0),   2)
        cv2.putText(vis, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return vis


# ─────────────────────────────────────────────────────────
# Open3D 3D 可视化（可选）
# ─────────────────────────────────────────────────────────

def visualize_3d(
    points_3d:   np.ndarray,
    obj_axes,
    grasp_geo=None,
    contact_pts: np.ndarray | None = None,
    window_name: str = "Object Pose",
) -> None:
    """用 Open3D 显示点云 + 主轴 + 接近方向 + 接触点。"""
    try:
        import open3d as o3d
    except ImportError:
        print("[visualize_3d] 未安装 open3d，跳过")
        return

    geoms = []

    # 物体点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.paint_uniform_color([0.6, 0.8, 0.6])
    geoms.append(pcd)

    # OBB 线框
    obb_ls = o3d.geometry.LineSet()
    obb_ls.points = o3d.utility.Vector3dVector(obj_axes.obb_vertices)
    obb_ls.lines  = o3d.utility.Vector2iVector([
        [0,1],[2,3],[4,5],[6,7],
        [0,2],[1,3],[4,6],[5,7],
        [0,4],[1,5],[2,6],[3,7],
    ])
    obb_ls.paint_uniform_color([1, 0.5, 0])
    geoms.append(obb_ls)

    # 主轴线段
    center = obj_axes.center
    for axis, half, color in [
        (obj_axes.long_axis,  obj_axes.long_size  / 2, [1, 0, 0]),
        (obj_axes.mid_axis,   obj_axes.mid_size   / 2, [0, 1, 0]),
        (obj_axes.short_axis, obj_axes.short_size / 2, [0, 0, 1]),
    ]:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector([center, center + axis * half])
        ls.lines  = o3d.utility.Vector2iVector([[0, 1]])
        ls.paint_uniform_color(color)
        geoms.append(ls)

    # 接近方向
    if grasp_geo is not None:
        w   = grasp_geo.wrist_pos
        tip = w + grasp_geo.approach_dir * 0.1
        ls  = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector([w, tip])
        ls.lines  = o3d.utility.Vector2iVector([[0, 1]])
        ls.paint_uniform_color([1, 0.5, 0])
        geoms.append(ls)

        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sph.translate(w)
        sph.paint_uniform_color([1, 0.8, 0])
        geoms.append(sph)

    # 接触点
    if contact_pts is not None:
        for cp in contact_pts:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
            sph.translate(cp)
            sph.paint_uniform_color([0, 1, 1])
            geoms.append(sph)

    o3d.visualization.draw_geometries(geoms, window_name=window_name)
