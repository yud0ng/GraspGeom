"""
object_analysis
================
物体姿态分析包：SAM3 文本分割 + MoGe2 深度 + PCA 姿态估计。

模块结构：
    pipeline.py        — 主流程入口（CLI）
    segmentor.py       — SAM3 文本概念分割
    depth_estimator.py — MoGe2 深度估计（含缓存）
    cloud_extractor.py — SAM3 mask + MoGe2 点图 → 物体点云
    pose_estimator.py  — PCA 主轴 + OBB 估计
    grasp_analyzer.py  — 手部-物体抓取几何分析
    hand_state.py      — VITRA hand_state.npy 加载与 FK
    visualizer.py      — 2D/3D 可视化工具

快速使用：
    python pipeline.py \\
        --video       path/to/video.mp4 \\
        --hand_state  path/to/video.hand_state.npy \\
        --object_name "scissors" \\
        --output_dir  ./results \\
        --save_video
"""

from .pose_estimator  import ObjectAxes, estimate_object_axes
from .grasp_analyzer  import (GraspGeometry, analyze_grasp_geometry,
                               estimate_contact_points)
from .hand_state      import (load_hand_state, get_available_frames,
                               get_camera_params, get_frame_data,
                               get_wrist_and_approach, get_hand_joints)
from .cloud_extractor import project_points, make_hand_mask, extract_object_cloud
from .depth_estimator import DepthEstimator
from .segmentor       import ObjectSegmentor
from .visualizer      import annotate_frame, visualize_3d

__all__ = [
    # 数据类
    'ObjectAxes',
    'GraspGeometry',
    # 姿态 & 抓取
    'estimate_object_axes',
    'analyze_grasp_geometry',
    'estimate_contact_points',
    # 手部状态
    'load_hand_state',
    'get_available_frames',
    'get_camera_params',
    'get_frame_data',
    'get_wrist_and_approach',
    'get_hand_joints',
    # 点云工具
    'project_points',
    'make_hand_mask',
    'extract_object_cloud',
    # 模型封装
    'DepthEstimator',
    'ObjectSegmentor',
    # 可视化
    'annotate_frame',
    'visualize_3d',
]
