"""
segmentor.py
=============
SAM3（Segment Anything Model 3）封装，使用文本概念提示分割物体。

SAM3 特性：
  - 输入自然语言描述（如 "scissors"、"cup"）直接分割所有匹配实例
  - 不需要手动点击或框选提示
  - 返回多个候选实例，本模块自动选取最靠近手腕的那个

依赖：
    pip install sam3
    # 或从 facebookresearch/sam3 源码安装
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# 本地已下载的 SAM3 检查点（从 facebook/sam3 下载）
_DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "facebook" / "sam3" / "sam3.pt"


class ObjectSegmentor:
    """
    SAM3 文本概念分割器。

    Args:
        checkpoint: SAM3 检查点路径，None 则使用默认（需联网或已下载）
        device:     'cuda' / 'cpu' / None（自动）
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str | None = None,
    ):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 确定检查点路径：优先用户指定，其次本地默认，最后 HuggingFace
        if checkpoint is None and _DEFAULT_CHECKPOINT.exists():
            checkpoint = str(_DEFAULT_CHECKPOINT)

        print(f"[ObjectSegmentor] 加载 SAM3 → {self.device}")
        print(f"[ObjectSegmentor] 检查点: {checkpoint or '从 HuggingFace 下载'}")

        model = build_sam3_image_model(
            checkpoint_path=checkpoint,
            load_from_HF=(checkpoint is None),
            device=self.device,
        )
        model = model.to(self.device)
        self.processor = Sam3Processor(model, device=self.device)

    def segment(
        self,
        frame_bgr: np.ndarray,
        object_name: str,
        wrist_uv: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """
        用文本概念提示分割物体，自动选取最靠近手腕的实例。

        Args:
            frame_bgr:   (H, W, 3) BGR 图像
            object_name: 物体名称，如 "scissors"、"cup"、"bottle"
            wrist_uv:    (2,) int，腕部在图像中的像素坐标（用于选实例）
                         None 则选得分最高的实例

        Returns:
            (H, W) uint8 mask（255=物体区域），若未检测到返回 None
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # addmm_act (perflib/fused.py) 强制将 fc1 输出转为 bfloat16，
        # 而 fc2.weight 仍是 float32 → dtype 不匹配。
        # autocast 会让标准 Linear ops 将 weight 也视为 bfloat16，消除冲突。
        with torch.autocast(
            device_type=self.device if self.device != 'cpu' else 'cpu',
            dtype=torch.bfloat16,
            enabled=(self.device == 'cuda'),
        ):
            state  = self.processor.set_image(pil_image)
            output = self.processor.set_text_prompt(state=state, prompt=object_name)

        masks  = output.get("masks")   # (N, 1, H, W) or (N, H, W)
        scores = output.get("scores")  # (N,) or None

        if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
            return None

        # ── 转为 (N, H, W) bool，先搬到 CPU ────────────
        if hasattr(masks, 'cpu'):
            masks = masks.cpu()
        masks = np.asarray(masks)
        if masks.ndim == 4:          # (N, 1, H, W) → (N, H, W)
            masks = masks[:, 0]
        if masks.dtype != bool:
            masks = masks > 0.5

        if masks.shape[0] == 0:
            return None

        if scores is not None and hasattr(scores, 'cpu'):
            scores = scores.cpu().float()
        scores_np = np.asarray(scores) if scores is not None else None
        return self._select_and_clean(masks, scores_np, wrist_uv)

    # ─────────────────────────────────────────────
    # 内部方法
    # ─────────────────────────────────────────────

    def _select_and_clean(
        self,
        masks: np.ndarray,       # (N, H, W) bool
        scores: np.ndarray | None,
        wrist_uv: np.ndarray | None,
    ) -> np.ndarray | None:
        """选出最佳实例并做形态学清理。"""
        best_idx = self._pick_best(masks, scores, wrist_uv)
        best = masks[best_idx].astype(np.uint8) * 255

        # 形态学：先闭运算填洞，再开运算去噪点
        k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        best = cv2.morphologyEx(best, cv2.MORPH_CLOSE, k11)
        best = cv2.morphologyEx(best, cv2.MORPH_OPEN,  k5)

        if best.sum() == 0:
            return None
        return best

    def _pick_best(
        self,
        masks: np.ndarray,
        scores: np.ndarray | None,
        wrist_uv: np.ndarray | None,
    ) -> int:
        """
        选取最佳实例索引。

        策略：
          1. 若提供 wrist_uv：选 mask 质心与腕部 2D 距离最小的实例
          2. 否则：选得分最高或面积最大的实例
        """
        N = masks.shape[0]

        if wrist_uv is not None:
            wx, wy = float(wrist_uv[0]), float(wrist_uv[1])
            best_idx, min_dist = 0, float('inf')
            for i in range(N):
                ys, xs = np.where(masks[i])
                if len(xs) == 0:
                    continue
                cx, cy = xs.mean(), ys.mean()
                d = (cx - wx) ** 2 + (cy - wy) ** 2
                if d < min_dist:
                    min_dist, best_idx = d, i
            return best_idx

        if scores is not None and len(scores) == N:
            return int(np.argmax(scores))

        # 按面积选最大
        return int(np.argmax([m.sum() for m in masks]))
