"""
pipeline.py
============
物体姿态分析主流程。

输入：
  --video        : 输入视频文件
  --hand_state   : VITRA 重建的 hand_state.npy
  --object_name  : 物体名称（SAM3 文本提示，如 "scissors"、"cup"）
  --output_dir   : 输出目录（默认 ./results）

可选：
  --stride       : 每隔几帧处理一次（默认 1）
  --depth_cache  : MoGe2 深度缓存目录（默认 <output_dir>/depths）
  --save_video   : 输出可视化 MP4
  --no_depth     : 跳过 MoGe2，只做 2D 分析（不推荐）
  --device       : cuda / cpu（默认自动）
  --sam_ckpt     : SAM3 检查点路径（可选，默认使用 sam3 默认权重）
  --min_points   : 点云最小点数（默认 30）

输出（保存到 output_dir）：
  vis/frame_XXXXX.jpg    每帧可视化图
  results.json           可序列化结果（不含点云）
  results.npy            完整结果（含点云、手部关节）
  analysis_vis.mp4       可视化视频（--save_video 时）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 将本目录加入 sys.path（在包外直接运行时用）
sys.path.insert(0, str(Path(__file__).parent))

from hand_state      import (load_hand_state, get_available_frames,
                              get_camera_params, get_frame_data,
                              get_wrist_and_approach, get_hand_joints)
from depth_estimator import DepthEstimator
from segmentor       import ObjectSegmentor
from cloud_extractor import project_points, make_hand_mask, extract_object_cloud
from pose_estimator  import estimate_object_axes
from grasp_analyzer  import analyze_grasp_geometry, estimate_contact_points
from visualizer      import annotate_frame


# ─────────────────────────────────────────────────────────
# 单帧处理
# ─────────────────────────────────────────────────────────

def process_frame(
    frame_bgr:   np.ndarray,
    frame_id:    int,
    frame_data:  dict,
    depth_est:   DepthEstimator,
    segmentor:   ObjectSegmentor,
    object_name: str,
    camera:      dict,
    fov_x:       float,
    min_points:  int,
) -> dict | None:
    """
    处理单帧，返回结果 dict 或 None（跳过该帧）。

    Result dict keys:
        frame_id, obj_axes, grasp_geo, contact_pts,
        points_3d, hand_joints, wrist_pos, approach_dir,
        object_mask, K, object_name
    """
    W, H = camera['width'], camera['height']

    # ── 1. 手部状态 ──────────────────────────────────
    wrist_pos, approach_dir = get_wrist_and_approach(frame_data)
    joints_3d = get_hand_joints(frame_data)

    # ── 2. MoGe2 深度 ────────────────────────────────
    depth_result = depth_est.infer(
        frame_bgr,
        fov_x_deg=fov_x if fov_x > 0 else None,
        frame_id=frame_id,
    )
    K = DepthEstimator.to_pixel_intrinsics(depth_result['intrinsics'], W, H)

    # ── 3. 腕部 2D 坐标（SAM3 实例选择提示） ─────────
    wrist_uv = project_points(wrist_pos[None], K)[0]

    # ── 4. SAM3 分割 ─────────────────────────────────
    object_mask = segmentor.segment(frame_bgr, object_name, wrist_uv=wrist_uv)
    if object_mask is None:
        print(f"  [frame {frame_id}] SAM3 未检测到 '{object_name}'，跳过")
        return None

    # ── 5. 手部排除 mask ─────────────────────────────
    hand_uv   = project_points(joints_3d, K)
    hand_mask = make_hand_mask((H, W), hand_uv, dilation=20)

    # ── 6. 提取物体点云 ──────────────────────────────
    points_3d = extract_object_cloud(
        depth_result['points'],
        depth_result['mask'],
        object_mask,
        hand_mask,
        min_points=min_points,
    )
    if points_3d is None:
        print(f"  [frame {frame_id}] 点云点数不足，跳过")
        return None

    # ── 7. 估计物体姿态 ──────────────────────────────
    obj_axes = estimate_object_axes(points_3d)
    if obj_axes is None:
        print(f"  [frame {frame_id}] PCA 失败，跳过")
        return None

    # ── 8. 抓取几何 ──────────────────────────────────
    grasp_geo   = analyze_grasp_geometry(obj_axes, wrist_pos, approach_dir)
    contact_pts = estimate_contact_points(obj_axes, approach_dir, n_candidates=4)

    return {
        'frame_id':    frame_id,
        'object_name': object_name,
        'obj_axes':    obj_axes,
        'grasp_geo':   grasp_geo,
        'contact_pts': contact_pts,
        'points_3d':   points_3d,
        'hand_joints': joints_3d,
        'wrist_pos':   wrist_pos,
        'approach_dir': approach_dir,
        'object_mask': object_mask,
        'K':           K,
    }


# ─────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="物体姿态分析：SAM3 分割 + MoGe2 深度 + PCA 姿态"
    )
    parser.add_argument('--video',       required=True,
                        help='输入视频路径')
    parser.add_argument('--hand_state',  required=True,
                        help='hand_state.npy 路径')
    parser.add_argument('--object_name', required=True,
                        help='物体名称，如 "scissors"、"cup"（SAM3 文本提示）')
    parser.add_argument('--output_dir',  default='./results',
                        help='输出目录（默认 ./results）')
    parser.add_argument('--stride',      type=int, default=1,
                        help='每隔几帧处理一次（默认 1）')
    parser.add_argument('--depth_cache', default=None,
                        help='MoGe2 深度缓存目录，None 则 <output_dir>/depths')
    parser.add_argument('--save_video',  action='store_true',
                        help='保存可视化 MP4')
    parser.add_argument('--no_depth',    action='store_true',
                        help='跳过 MoGe2（实验性，不推荐）')
    parser.add_argument('--device',      default=None,
                        help='cuda / cpu（默认自动选择）')
    parser.add_argument('--sam_ckpt',    default=None,
                        help='SAM3 检查点路径（可选）')
    parser.add_argument('--min_points',  type=int, default=30,
                        help='点云最小点数（默认 30）')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    vis_dir    = output_dir / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

    depth_cache = Path(args.depth_cache) if args.depth_cache \
                  else output_dir / 'depths'

    # ── 加载 hand_state ──────────────────────────────
    print(f"加载 hand_state: {args.hand_state}")
    hand_state = load_hand_state(args.hand_state)
    camera     = get_camera_params(hand_state)
    fov_x      = camera['fov_x']
    W, H       = camera['width'], camera['height']
    fps        = camera['fps']

    all_frame_ids = get_available_frames(hand_state)
    frame_ids     = all_frame_ids[::args.stride]
    print(f"共 {len(all_frame_ids)} 帧有重建结果，按 stride={args.stride} 处理 {len(frame_ids)} 帧")
    print(f"物体名称: '{args.object_name}'  分辨率: {W}×{H}  FOV: {fov_x:.1f}°")

    # ── 加载模型 ─────────────────────────────────────
    depth_est = DepthEstimator(
        device=args.device,
        cache_dir=depth_cache,
    )
    segmentor = ObjectSegmentor(
        checkpoint=args.sam_ckpt,
        device=args.device,
    )

    # ── 打开视频 ─────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    # ── 可选：视频写入器 ─────────────────────────────
    video_writer = None
    if args.save_video:
        out_mp4 = str(output_dir / 'analysis_vis.mp4')
        fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_mp4, fourcc, fps, (W, H))

    # ── 逐帧处理 ─────────────────────────────────────
    all_results  = []
    success_cnt  = 0

    for i, frame_id in enumerate(frame_ids):
        frame_data = get_frame_data(hand_state, frame_id)
        if frame_data is None:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"  [frame {frame_id}] 读取失败，跳过")
            continue

        result = process_frame(
            frame_bgr   = frame_bgr,
            frame_id    = frame_id,
            frame_data  = frame_data,
            depth_est   = depth_est,
            segmentor   = segmentor,
            object_name = args.object_name,
            camera      = camera,
            fov_x       = fov_x,
            min_points  = args.min_points,
        )

        if result is None:
            continue

        success_cnt += 1

        # ── 可视化 ──────────────────────────────────
        vis = annotate_frame(
            frame       = frame_bgr,
            obj_axes    = result['obj_axes'],
            grasp_geo   = result['grasp_geo'],
            contact_pts = result['contact_pts'],
            obj_mask    = result['object_mask'],
            K           = result['K'],
            frame_id    = frame_id,
            object_name = args.object_name,
        )

        # 保存 JPEG
        jpg_path = vis_dir / f'frame_{frame_id:05d}.jpg'
        cv2.imwrite(str(jpg_path), vis)

        if video_writer is not None:
            video_writer.write(vis)

        # 进度
        if (i + 1) % 20 == 0 or (i + 1) == len(frame_ids):
            print(f"  进度: {i+1}/{len(frame_ids)}  成功: {success_cnt}"
                  f"  当前帧: {frame_id}")

        # 积累结果（JSON 可序列化部分）
        all_results.append({
            'frame_id':    frame_id,
            'object_name': args.object_name,
            'obj_axes':    result['obj_axes'].to_dict(),
            'grasp_geo':   result['grasp_geo'].to_dict(),
            'contact_pts': result['contact_pts'].tolist(),
            'n_points':    int(len(result['points_3d'])),
        })

    cap.release()
    if video_writer is not None:
        video_writer.release()

    # ── 保存结果 ─────────────────────────────────────
    json_path = output_dir / 'results.json'
    with open(str(json_path), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存：{json_path}")

    # NPY（含点云，文件较大）
    npy_path = output_dir / 'results.npy'
    np.save(str(npy_path), all_results, allow_pickle=True)
    print(f"JSON 摘要：{json_path}")
    print(f"NPY  详情：{npy_path}")
    if args.save_video:
        print(f"可视化视频：{output_dir / 'analysis_vis.mp4'}")
    print(f"\n完成！处理 {len(frame_ids)} 帧，成功 {success_cnt} 帧。")


if __name__ == '__main__':
    main()
