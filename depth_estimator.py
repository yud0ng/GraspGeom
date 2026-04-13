"""
depth_estimator.py
===================
MoGe2 深度估计封装，支持逐帧推理和 NPZ 缓存。

输出格式（每帧）：
    depth      : (H, W) float32，metric depth（米）
    points     : (H, W, 3) float32，3D 点图（相机坐标系）
    mask       : (H, W) bool，有效像素
    intrinsics : (3, 3) float32，归一化内参
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# 将 VITRA 加入 sys.path 以使用 MoGe
_VITRA_ROOT = Path(__file__).parent.parent / 'VITRA'


class DepthEstimator:
    """
    MoGe2 深度估计器，支持可选 NPZ 磁盘缓存。

    Args:
        model_name:  HuggingFace 模型 ID，默认 "Ruicheng/moge-2-vitl"
        device:      torch.device，None 则自动选择
        cache_dir:   NPZ 缓存目录，None 则不缓存
    """

    def __init__(
        self,
        model_name: str = "Ruicheng/moge-2-vitl",
        device: torch.device | None = None,
        cache_dir: str | Path | None = None,
    ):
        if str(_VITRA_ROOT) not in sys.path:
            sys.path.insert(0, str(_VITRA_ROOT))

        from thirdparty.MoGe.moge.model.v2 import MoGeModel as MoGeModelV2

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[DepthEstimator] 加载 {model_name} → {self.device}")
        self.model = MoGeModelV2.from_pretrained(model_name).to(self.device).eval()
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @torch.inference_mode()
    def infer(
        self,
        frame_bgr: np.ndarray,
        fov_x_deg: float | None = None,
        resolution_level: int = 9,
        frame_id: int | None = None,
    ) -> dict:
        """
        对单帧运行 MoGe2，返回深度 + 3D 点图。

        优先从磁盘缓存读取，若无缓存则推理并写入缓存。

        Args:
            frame_bgr:        (H, W, 3) BGR 图像
            fov_x_deg:        水平 FOV（度），None 由模型估计
            resolution_level: 分辨率级别（9=最高质量）
            frame_id:         帧号，用于缓存文件名

        Returns:
            dict:
                depth      (H, W) float32
                points     (H, W, 3) float32
                mask       (H, W) bool
                intrinsics (3, 3) float32，归一化
        """
        # ── 尝试读缓存 ───────────────────────────────
        if self.cache_dir is not None and frame_id is not None:
            cache_path = self.cache_dir / f"frame_{frame_id:05d}.npz"
            if cache_path.exists():
                data = np.load(str(cache_path))
                return {
                    'depth':      data['depth'],
                    'points':     data['points'],
                    'mask':       data['mask'].astype(bool),
                    'intrinsics': data['intrinsics'],
                }

        # ── 推理 ─────────────────────────────────────
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.tensor(
            frame_rgb / 255.0, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)   # (3, H, W)

        kwargs: dict = {'resolution_level': resolution_level}
        if fov_x_deg is not None and fov_x_deg > 0:
            kwargs['fov_x'] = fov_x_deg

        out = self.model.infer(img_t, **kwargs)

        result = {
            'depth':      out['depth'].cpu().numpy().astype(np.float32),
            'points':     out['points'].cpu().numpy().astype(np.float32),
            'mask':       out['mask'].cpu().numpy().astype(bool),
            'intrinsics': out['intrinsics'].cpu().numpy().astype(np.float32),
        }

        # ── 写缓存 ───────────────────────────────────
        if self.cache_dir is not None and frame_id is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(self.cache_dir / f"frame_{frame_id:05d}.npz"),
                **result,
            )

        return result

    @staticmethod
    def to_pixel_intrinsics(
        intrinsics: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """
        将 MoGe2 归一化内参转换为像素空间 K 矩阵 (3×3)。

        MoGe2 归一化约定：
            intrinsics[0, :] 已被 width  归一化
            intrinsics[1, :] 已被 height 归一化
        """
        K = intrinsics.copy()
        K[0] *= width
        K[1] *= height
        return K
