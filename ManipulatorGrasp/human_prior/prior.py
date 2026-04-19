"""
GraspPrior dataclass — output of the human video analysis pipeline.

All 3D quantities are expressed in the demo video camera frame
(X right, Y down, Z into scene) unless noted otherwise.

OBB (oriented bounding box) fields come from GraspGeom PCA and are the
primary reranking signal:
  - face_offset_obj_frame is camera-frame-independent (expressed in the
    object's own principal-axis frame), so it transfers directly to the
    robot camera at execution time without any explicit camera calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class GraspPrior:
    """Human-demonstration-derived prior for a single grasp interaction.

    Required fields (Phase-1 geometric reranking):
        contact_point_3d   — where the human fingers made contact
        approach_direction — direction the wrist moved toward the object

    GraspGeom OBB fields (Phase-2 OBB-guided reranking):
        obb_axes             — (3,3) principal-axis matrix, demo cam frame.
                               Columns are [long_axis, mid_axis, short_axis].
        obb_sizes            — (3,) [long, mid, short] dimensions in metres
        face_offset_obj_frame — (3,) target face-center offset in OBB coords.
                               Camera-frame-independent: at execution time,
                               robot_contact = robot_centroid + R_robot @ face_offset_obj_frame
                               where R_robot is the robot-cam OBB axis matrix.

    Runtime-only (not persisted to disk):
        robot_obb_axes — (3,3) OBB axes in robot camera frame.
                         Set by resolve_prior_to_robot_frame() in main.py.
    """

    # --- required ---
    contact_point_3d: np.ndarray       # (3,)
    approach_direction: np.ndarray     # (3,) unit vector

    # --- object-relative contact (simple fallback when OBB not available) ---
    contact_point_relative: Optional[np.ndarray] = None   # (3,) offset from centroid
    object_centroid_3d: Optional[np.ndarray] = None       # (3,) in demo cam frame

    # --- GraspGeom OBB fields (primary alignment signal) ---
    obb_axes: Optional[np.ndarray] = None              # (3, 3) cols=[long,mid,short]
    obb_sizes: Optional[np.ndarray] = None             # (3,) in demo depth units (not metres)
    face_offset_normalized: Optional[np.ndarray] = None  # (3,) dimensionless: offset / obb_sizes
    # face_offset_normalized is scale-invariant:
    #   values like [0, -0.5, 0] mean "half of the mid-axis, negative side"
    #   At robot time: face_offset_robot = face_offset_normalized * robot_obb_sizes (metres)
    #                  robot_contact     = robot_centroid + R_robot @ face_offset_robot

    # --- runtime-only: populated by resolve_prior_to_robot_frame(), not saved ---
    robot_obb_axes: Optional[np.ndarray] = None        # (3, 3) robot cam frame

    # --- optional richer prior ---
    wrist_pose: Optional[np.ndarray] = None      # (6,) [tx, ty, tz, rx, ry, rz]
    finger_config: Optional[np.ndarray] = None   # (45,) MANO joint angles
    object_class: Optional[str] = None
    object_embedding: Optional[np.ndarray] = None  # CLIP (512,)
    object_mask: Optional[np.ndarray] = None     # (H, W) bool
    contact_heatmap: Optional[np.ndarray] = None # (H, W) float
    grasp_event: Optional[Tuple[int, int]] = None  # (start_frame, end_frame)

    def __post_init__(self):
        self.contact_point_3d = np.asarray(self.contact_point_3d, dtype=np.float32)
        self.approach_direction = np.asarray(self.approach_direction, dtype=np.float32)
        norm = np.linalg.norm(self.approach_direction)
        if norm > 1e-6:
            self.approach_direction = self.approach_direction / norm

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save prior to a .npz + sidecar .txt.  robot_obb_axes is not saved."""
        path = Path(path)
        arrays: dict = dict(
            contact_point_3d=self.contact_point_3d,
            approach_direction=self.approach_direction,
        )
        for key in ("contact_point_relative", "object_centroid_3d",
                    "obb_axes", "obb_sizes", "face_offset_normalized",
                    "wrist_pose", "finger_config", "object_embedding"):
            val = getattr(self, key)
            if val is not None:
                arrays[key] = np.asarray(val, dtype=np.float32)
        if self.object_mask is not None:
            arrays["object_mask"] = self.object_mask.astype(np.bool_)
        if self.contact_heatmap is not None:
            arrays["contact_heatmap"] = self.contact_heatmap.astype(np.float32)
        if self.grasp_event is not None:
            arrays["grasp_event"] = np.array(self.grasp_event, dtype=np.int32)

        np.savez_compressed(str(path), **arrays)

        meta_path = path.with_suffix(".txt")
        with open(meta_path, "w") as f:
            f.write(f"object_class={self.object_class or ''}\n")

        print(f"[GraspPrior] saved → {path}  +  {meta_path}")

    @classmethod
    def load(cls, path: str) -> "GraspPrior":
        """Load a prior saved with .save()."""
        path = Path(path)
        npz_path = path.with_suffix(".npz") if path.suffix != ".npz" else path
        data = np.load(str(npz_path), allow_pickle=False)

        kwargs: dict = dict(
            contact_point_3d=data["contact_point_3d"],
            approach_direction=data["approach_direction"],
        )
        for key in ("contact_point_relative", "object_centroid_3d",
                    "obb_axes", "obb_sizes", "face_offset_normalized",
                    "wrist_pose", "finger_config", "object_embedding",
                    "object_mask", "contact_heatmap"):
            if key in data:
                kwargs[key] = data[key]
        if "grasp_event" in data:
            kwargs["grasp_event"] = tuple(int(x) for x in data["grasp_event"])

        meta_path = npz_path.with_suffix(".txt")
        if meta_path.exists():
            for line in meta_path.read_text().splitlines():
                k, _, v = line.partition("=")
                if k == "object_class":
                    kwargs["object_class"] = v or None

        return cls(**kwargs)

    def summary(self) -> str:
        has_obb = self.face_offset_normalized is not None
        obb_str = (f"set  sizes={self.obb_sizes.round(3)}"
                   if has_obb and self.obb_sizes is not None else "not set")
        lines = [
            "GraspPrior:",
            f"  object_class          : {self.object_class}",
            f"  contact_point_3d      : {self.contact_point_3d.round(4)}",
            f"  contact_point_relative: {self.contact_point_relative.round(4) if self.contact_point_relative is not None else None}",
            f"  object_centroid_3d    : {self.object_centroid_3d.round(4) if self.object_centroid_3d is not None else None}",
            f"  approach_direction    : {self.approach_direction.round(4)}",
            f"  obb                   : {obb_str}",
            f"  face_offset_normalized: {self.face_offset_normalized.round(4) if has_obb else None}",
            f"  grasp_event           : {self.grasp_event}",
            f"  has wrist_pose        : {self.wrist_pose is not None}",
            f"  has object_emb        : {self.object_embedding is not None}",
            f"  has object_mask       : {self.object_mask is not None}",
        ]
        return "\n".join(lines)
