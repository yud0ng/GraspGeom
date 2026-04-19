"""
Human-prior extraction pipeline: egocentric video → GraspPrior

Run this script in the `vitra` conda env (Python 3.10):

    cd /media/yutao/T91/projects/Manipulator_Grasp
    /media/yutao/T91/miniconda3/envs/vitra/bin/python -m human_prior.extract \
        --video /path/to/demo.mp4 \
        --object "mug" \
        --output priors/mug_demo

The saved .npz files are then loaded by the graspnet env at robot execution time.

---
Pipeline stages:
  1. HaWoR     — hand reconstruction → MANO joints (21 keypoints, camera space)
                 Fallback: MediaPipe (simpler, no MANO license needed)
  2. Action segmentation — wrist speed minima → grasp event windows
  3. Grounded-SAM2 — object mask per contact frame
  4. Depth Anything V2 — depth map → lift fingertips to 3D
  5. CLIP       — object embedding for cross-view matching

NOTE: HaWoR requires MANO model weights (manual download after registering at
      https://mano.is.tue.mpg.de). See setup instructions at bottom of file.

All 3D coordinates are in the video camera frame (X right, Y down, Z forward).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

HAWOR_ROOT     = Path("/media/yutao/T91/projects/VITRA/thirdparty/HaWoR")
GSAM2_ROOT     = Path("/media/yutao/T91/projects/Grounded-SAM-2")
DEPTH_ROOT     = Path("/media/yutao/T91/projects/Depth-Anything-V2")
GRASPGEOM_ROOT = Path("/media/yutao/T91/projects/GraspGeom")
VITRA_ENV_BIN  = Path("/media/yutao/T91/miniconda3/envs/vitra/bin")

# Ensure the vitra env's bin (ffmpeg, etc.) is on PATH when running subprocesses
os.environ["PATH"] = str(VITRA_ENV_BIN) + ":" + os.environ.get("PATH", "")

# SAM2 config is inside the sam2 package, not at the repo root
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ---------------------------------------------------------------------------
# Stage 1a — HaWoR hand reconstruction
# ---------------------------------------------------------------------------

def track_hands_hawor(
    video_path: str,
    checkpoint: str = str(HAWOR_ROOT / "weights/hawor/checkpoints/hawor.ckpt"),
    infiller_weight: str = str(HAWOR_ROOT / "weights/hawor/checkpoints/infiller.pt"),
) -> Tuple[List[Optional[np.ndarray]], float]:
    """
    Run HaWoR on a video to get per-frame MANO hand joints in camera space.

    Args:
        video_path       : path to input video.
        checkpoint       : HaWoR model checkpoint.
        infiller_weight  : HaWoR infiller checkpoint.

    Returns:
        joint_sequence : List[np.ndarray (21, 3) | None], one per video frame.
                         None where no hand was detected.
                         Indices: 0=wrist, 4=thumb tip, 8=index tip,
                                  12=middle tip, 16=ring tip, 20=pinky tip.
        fps            : float, video frame rate.

    MANO required: place MANO_RIGHT.pkl at
        {HAWOR_ROOT}/_DATA/data/mano/MANO_RIGHT.pkl
    Register at https://mano.is.tue.mpg.de to download.
    """
    import torch

    # Add HaWoR to path
    for p in [str(HAWOR_ROOT), str(HAWOR_ROOT / "lib")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # HaWoR loads detector/checkpoint via relative paths — must run from HAWOR_ROOT
    _orig_cwd = os.getcwd()
    os.chdir(str(HAWOR_ROOT))

    import types
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation
    from hawor.utils.process import run_mano

    # Build minimal args namespace that HaWoR expects
    args = types.SimpleNamespace(
        video_path=video_path,
        input_type="file",
        checkpoint=checkpoint,
        infiller_weight=infiller_weight,
        img_focal=None,
        vis_mode="cam",
    )

    # --- Step 1: detect & track hands across frames ---
    start_idx, end_idx, seq_folder, _ = detect_track_video(args)

    # --- Step 2: camera-space hand motion estimation (no DROID-SLAM needed) ---
    frame_chunks_all, _ = hawor_motion_estimation(
        args, start_idx, end_idx, seq_folder
    )

    # --- Decode per-frame MANO joints ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    joint_sequence: List[Optional[np.ndarray]] = [None] * n_frames

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for chunk in frame_chunks_all:
        is_right = chunk.get("is_right", True)
        if not is_right:
            continue   # only use right hand

        frame_ids     = chunk["frame_ids"]       # (T,)
        global_orient = chunk["global_orient"]   # (T, 3) axis-angle
        hand_pose     = chunk["hand_pose"]       # (T, 45) MANO pose params
        betas         = chunk["betas"]           # (T, 10) shape params
        _             = chunk["pred_cam"]        # (T, 3) weak-persp — not used

        # Run MANO forward to get 3D joints in camera space
        joints_batch = run_mano(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            is_right=is_right,
        )                                         # (T, 21, 3) in MANO space

        # Translate to approximate camera-frame coordinates using pred_cam
        # pred_cam = [scale, tx, ty]; z derived from scale
        for i, fid in enumerate(frame_ids):
            if fid < n_frames:
                joint_sequence[int(fid)] = joints_batch[i]  # (21, 3)

    os.chdir(_orig_cwd)   # restore working directory
    print(f"[HaWoR] Detected hands in "
          f"{sum(1 for j in joint_sequence if j is not None)}/{n_frames} frames")
    return joint_sequence, fps


# ---------------------------------------------------------------------------
# Stage 1b — MediaPipe fallback (no MANO license needed)
# ---------------------------------------------------------------------------

def track_hands_mediapipe(
    frames: List[np.ndarray],
    camera_K: np.ndarray,
) -> List[Optional[np.ndarray]]:
    """
    Fallback hand tracker using MediaPipe Tasks API (v0.10+).
    Same 21-landmark output format as HaWoR.
    Use this if MANO weights are not yet available.

    Requires models/hand_landmarker.task (downloaded by setup_hawor_weights.sh).
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, RunningMode,
    )

    model_path = str(Path(__file__).parent.parent / "models" / "hand_landmarker.task")
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"MediaPipe model not found at {model_path}.\n"
            "Download with:\n"
            "  wget -O models/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    H, W = frames[0].shape[:2]
    fx, fy = camera_K[0, 0], camera_K[1, 1]
    cx, cy = camera_K[0, 2], camera_K[1, 2]
    WRIST_DEPTH_M = 0.4   # assumed wrist depth from camera (egocentric)

    results_list: List[Optional[np.ndarray]] = []

    with HandLandmarker.create_from_options(options) as detector:
        for frame in frames:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection = detector.detect(mp_image)

            if not detection.hand_landmarks:
                results_list.append(None)
                continue

            # Only use the right hand
            hand_idx = None
            for i, handedness in enumerate(detection.handedness):
                if handedness[0].category_name == "Right":
                    hand_idx = i
                    break
            if hand_idx is None:
                results_list.append(None)
                continue

            lms       = detection.hand_landmarks[hand_idx]        # 21 NormalizedLandmark
            world_lms = detection.hand_world_landmarks[hand_idx]  # 21 Landmark (metres)

            wrist_world_z = world_lms[0].z
            joints = np.zeros((21, 3), dtype=np.float32)
            for j in range(21):
                u = lms[j].x * W
                v = lms[j].y * H
                z_abs = max(WRIST_DEPTH_M + (world_lms[j].z - wrist_world_z), 0.05)
                joints[j] = [(u - cx) * z_abs / fx,
                             (v - cy) * z_abs / fy,
                             z_abs]
            results_list.append(joints)

    return results_list


def load_frames_from_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


# ---------------------------------------------------------------------------
# Stage 2 — Atomic action segmentation (wrist speed minima)
# ---------------------------------------------------------------------------

def segment_grasp_events(
    joint_sequence: List[Optional[np.ndarray]],
    fps: float,
    speed_threshold_percentile: float = 30.0,
    min_segment_frames: int = 3,
) -> List[Tuple[int, int]]:
    """
    Detect grasp events as low-wrist-speed intervals (ViTRA Stage 2 logic).

    Returns list of (start_frame, end_frame) tuples.
    Falls back to the longest contiguous hand-detected window if no slow
    segment passes the length threshold (common with short demo clips).
    """
    wrist_positions, valid_indices = [], []
    for i, joints in enumerate(joint_sequence):
        if joints is not None:
            wrist_positions.append(joints[0])
            valid_indices.append(i)

    if len(wrist_positions) < 3:
        return []

    positions = np.array(wrist_positions)
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    threshold = np.percentile(speeds, speed_threshold_percentile)

    is_slow = speeds < threshold
    segments, in_seg, seg_start = [], False, 0
    for i, slow in enumerate(is_slow):
        if slow and not in_seg:
            in_seg, seg_start = True, i
        elif not slow and in_seg:
            in_seg = False
            if i - seg_start >= min_segment_frames:
                segments.append((valid_indices[seg_start],
                                  valid_indices[min(i, len(valid_indices)-1)]))
    if in_seg and len(is_slow) - seg_start >= min_segment_frames:
        segments.append((valid_indices[seg_start], valid_indices[-1]))

    if segments:
        return segments

    # Fallback: no slow segment found — use the longest contiguous hand window
    print("[segment_grasp_events] No slow segment found; "
          "using longest contiguous hand-detected window as fallback.")
    runs, run_start = [], valid_indices[0]
    prev = valid_indices[0]
    for idx in valid_indices[1:]:
        if idx > prev + 5:   # gap > 5 frames → new run
            runs.append((run_start, prev))
            run_start = idx
        prev = idx
    runs.append((run_start, prev))
    # return the longest run
    return [max(runs, key=lambda r: r[1] - r[0])]


# ---------------------------------------------------------------------------
# Stage 3 — Grounded-SAM2 object segmentation
# ---------------------------------------------------------------------------

def segment_object_gsam2(
    frame: np.ndarray,
    text_prompt: str,
    sam2_checkpoint: str = str(GSAM2_ROOT / "checkpoints/sam2.1_hiera_large.pt"),
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    grounding_dino_id: str = "IDEA-Research/grounding-dino-tiny",
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> Optional[np.ndarray]:
    """
    Segment a named object using Grounded-SAM2.
    Returns (H, W) bool mask or None if not found.

    SAM2 checkpoint: download with
        cd /media/yutao/T91/projects/Grounded-SAM-2
        bash download_ckpts.sh
    or the wget in setup_hawor_weights.sh.
    """
    import torch
    from PIL import Image as PILImage
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    if str(GSAM2_ROOT) not in sys.path:
        sys.path.insert(0, str(GSAM2_ROOT))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Grounding DINO detection
    dino_proc  = AutoProcessor.from_pretrained(grounding_dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        grounding_dino_id).to(device)

    pil_image = PILImage.fromarray(frame)
    # Grounding DINO requires text prompt to end with a period
    dino_prompt = text_prompt if text_prompt.endswith(".") else text_prompt + "."
    inputs = dino_proc(images=pil_image, text=dino_prompt,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[pil_image.size[::-1]],
    )[0]

    if len(results["boxes"]) == 0:
        print(f"  [SAM2] No '{text_prompt}' found in frame.")
        return None

    box = results["boxes"][results["scores"].argmax().item()].cpu().numpy()

    # SAM2 refinement
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    predictor  = SAM2ImagePredictor(sam2_model)
    predictor.set_image(frame)
    masks, _, _ = predictor.predict(box=box[None], multimask_output=False)
    return masks[0].astype(bool)


# ---------------------------------------------------------------------------
# Stage 4 — Depth Anything V2
# ---------------------------------------------------------------------------

def estimate_depth(
    frame: np.ndarray,
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
) -> np.ndarray:
    """
    Returns (H, W) float32 relative depth via Depth Anything V2 (HuggingFace).
    Scale-align with align_depth_scale() if robot RGB-D is available.
    """
    import torch
    from transformers import pipeline as hf_pipeline
    from PIL import Image as PILImage

    device = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline("depth-estimation", model=model_name, device=device)
    depth = np.array(pipe(PILImage.fromarray(frame))["depth"], dtype=np.float32)
    if depth.shape != frame.shape[:2]:
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    return depth


def align_depth_scale(
    pseudo: np.ndarray, metric: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    valid = mask & (metric > 0) & (pseudo > 0)
    if valid.sum() < 10:
        return pseudo
    return pseudo * (np.median(metric[valid]) / np.median(pseudo[valid]))


# ---------------------------------------------------------------------------
# GraspGeom — OBB estimation from object point cloud
# ---------------------------------------------------------------------------

def run_graspgeom_pca(
    obj_pts3d: np.ndarray,
    approach_dir: np.ndarray,
    obj_centroid: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Run GraspGeom PCA on the object point cloud to get OBB fields.

    Args:
        obj_pts3d    : (N, 3) 3D object points in demo camera frame.
        approach_dir : (3,) wrist approach direction (unit vector).
        obj_centroid : (3,) object centroid used to compute relative offset.

    Returns:
        obb_axes             : (3, 3) cols=[long,mid,short] in demo cam frame, or None
        obb_sizes            : (3,) [long, mid, short] in metres, or None
        face_offset_obj_frame: (3,) target face-center offset in OBB coordinate frame
                               (camera-independent), or None.
    """
    if str(GRASPGEOM_ROOT) not in sys.path:
        sys.path.insert(0, str(GRASPGEOM_ROOT))

    try:
        from pose_estimator import estimate_object_axes
        from grasp_analyzer import estimate_contact_points
    except ImportError as e:
        print(f"  [GraspGeom] import failed ({e}), skipping OBB.")
        return None, None, None

    obj_axes = estimate_object_axes(obj_pts3d)
    if obj_axes is None:
        print("  [GraspGeom] PCA failed (too few points), skipping OBB.")
        return None, None, None

    # Best contact face: OBB face whose normal is most anti-aligned with approach_dir
    # (estimate_contact_points returns face centers ranked by this criterion)
    contact_candidates = estimate_contact_points(obj_axes, approach_dir, n_candidates=1)
    best_face_center = contact_candidates[0]  # (3,) in demo cam frame

    # Offset from object centroid in demo cam frame
    face_center_rel_cam = best_face_center - obj_centroid  # (3,)

    # Convert to OBB coordinate frame (camera-independent)
    # R_demo cols = [long, mid, short]; R_demo.T projects cam-frame → OBB-frame
    R_demo = obj_axes.axis_matrix()  # (3, 3)
    face_offset_raw = (R_demo.T @ face_center_rel_cam).astype(np.float32)

    obb_axes  = R_demo.astype(np.float32)
    obb_sizes = np.array([obj_axes.long_size, obj_axes.mid_size, obj_axes.short_size],
                         dtype=np.float32)

    # Normalise by OBB sizes → dimensionless [~-0.5 .. 0.5 per axis]
    # Depth Anything gives relative depth (not metres), so raw offset is unscaled.
    # face_offset_normalized is scale-invariant: at robot time multiply by robot OBB sizes (metres).
    face_offset_normalized = face_offset_raw / (obb_sizes + 1e-9)

    print(f"  [GraspGeom] OBB sizes (L/M/S) : {obb_sizes.round(3)} (depth units)")
    print(f"  [GraspGeom] face_offset_norm   : {face_offset_normalized.round(4)}  (dimensionless)")
    return obb_axes, obb_sizes, face_offset_normalized


# ---------------------------------------------------------------------------
# Stage 5 — CLIP embedding
# ---------------------------------------------------------------------------

def compute_clip_embedding(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Returns (512,) CLIP ViT-B/32 embedding of the masked object crop."""
    import torch, clip
    from PIL import Image as PILImage

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.zeros(512, dtype=np.float32)
    crop = PILImage.fromarray(
        frame[rows.min():rows.max()+1, cols.min():cols.max()+1]
    )
    with torch.no_grad():
        emb = model.encode_image(preprocess(crop).unsqueeze(0).to(device))
    return emb[0].cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def project_to_image(pts3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = pts3d[:, 2:3].clip(1e-6, None)
    return (pts3d[:, :2] / z) * K[[0,1],[0,1]] + K[[0,1],[2,2]]


def lift_to_3d(px: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    c = px[:, 0].astype(int).clip(0, depth.shape[1]-1)
    r = px[:, 1].astype(int).clip(0, depth.shape[0]-1)
    z = depth[r, c].astype(np.float32)
    return np.stack([(c - K[0,2]) * z / K[0,0],
                     (r - K[1,2]) * z / K[1,1], z], axis=1)


def _build_heatmap(pts2d: np.ndarray, H: int, W: int, sigma: float = 5.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    hm = np.zeros((H, W), dtype=np.float32)
    for uv in pts2d:
        hm[np.clip(int(round(uv[1])), 0, H-1),
           np.clip(int(round(uv[0])), 0, W-1)] += 1.0
    hm = gaussian_filter(hm, sigma=sigma)
    return hm / hm.max() if hm.max() > 0 else hm


# ---------------------------------------------------------------------------
# Coordinate frame alignment (human cam → robot RGB-D cam)
# ---------------------------------------------------------------------------

def align_to_robot_frame(
    contact_point: np.ndarray,
    approach_direction: np.ndarray,
    human_cloud: np.ndarray,
    robot_cloud: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """ICP-align prior from human camera frame to robot RGB-D frame."""
    import open3d as o3d

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(human_cloud)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(robot_cloud)

    T_init = np.eye(4)
    T_init[:3, 3] = np.mean(robot_cloud, 0) - np.mean(human_cloud, 0)
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, max_correspondence_distance=0.05, init=T_init,
        estimation_method=o3d.pipelines.registration
            .TransformationEstimationPointToPoint(),
    )
    R, t = result.transformation[:3, :3], result.transformation[:3, 3]
    new_cp = R @ contact_point + t
    new_ap = R @ approach_direction
    return new_cp, new_ap / (np.linalg.norm(new_ap) + 1e-8)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def extract_prior_from_video(
    video_path: str,
    object_name: str,
    camera_K: Optional[np.ndarray] = None,
    robot_depth: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    use_hawor: bool = True,
    hawor_checkpoint: str = str(HAWOR_ROOT / "weights/hawor/checkpoints/hawor.ckpt"),
    hawor_infiller:   str = str(HAWOR_ROOT / "weights/hawor/checkpoints/infiller.pt"),
    sam2_checkpoint:  str = str(GSAM2_ROOT / "checkpoints/sam2.1_hiera_large.pt"),
    sam2_config:      str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    depth_model:      str = "depth-anything/Depth-Anything-V2-Small-hf",
) -> List["GraspPrior"]:
    """
    Egocentric video + object name → list of GraspPrior objects.

    Args:
        video_path    : path to demonstration video.
        object_name   : text label, e.g. "mug", "red bottle".
        camera_K      : (3, 3) intrinsics. Estimated from 60° FoV if None.
        robot_depth   : (H, W) metric depth from robot RGB-D for scale alignment.
        output_path   : stem for saving .npz files (e.g. "priors/mug").
        use_hawor     : True → HaWoR (MANO, accurate); False → MediaPipe (fallback).

    Returns:
        List of GraspPrior, one per detected grasp event.
    """
    from .prior import GraspPrior

    frames, fps = load_frames_from_video(video_path)
    H, W = frames[0].shape[:2]
    print(f"[extract] Loaded {len(frames)} frames @ {fps:.1f} fps")

    # Camera intrinsics
    if camera_K is None:
        fx = W / (2.0 * np.tan(np.deg2rad(30.0)))   # assume 60° horizontal FoV
        camera_K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float32)
        print(f"[extract] Estimated K: fx={fx:.1f}")

    # Stage 1: hand tracking
    if use_hawor:
        print("[extract] Running HaWoR hand reconstruction...")
        try:
            joint_sequence, _ = track_hands_hawor(
                video_path, hawor_checkpoint, hawor_infiller
            )
        except Exception as e:
            print(f"[extract] HaWoR failed ({e}), falling back to MediaPipe.")
            joint_sequence = track_hands_mediapipe(frames, camera_K)
    else:
        print("[extract] Running MediaPipe hand tracking...")
        joint_sequence = track_hands_mediapipe(frames, camera_K)

    detected = sum(1 for j in joint_sequence if j is not None)
    print(f"[extract] Hand detected in {detected}/{len(frames)} frames")

    # Stage 2: grasp event segmentation
    grasp_events = segment_grasp_events(joint_sequence, fps)
    print(f"[extract] {len(grasp_events)} grasp event(s): {grasp_events}")

    priors: List[GraspPrior] = []

    for ev_idx, (start_f, end_f) in enumerate(grasp_events):
        print(f"\n[extract] Event {ev_idx+1}: frames {start_f}–{end_f}")

        # Contact frame = last frame with hand detected
        cf = end_f
        while cf >= start_f and joint_sequence[cf] is None:
            cf -= 1
        if cf < start_f:
            print("  No hand detected in contact window, skipping.")
            continue

        frame_rgb  = frames[cf]
        joints_3d  = joint_sequence[cf]   # (21, 3)

        # Fingertips: indices 4, 8, 12, 16, 20
        ft3d = joints_3d[[4, 8, 12, 16, 20]]   # (5, 3)
        ft2d = project_to_image(ft3d, camera_K)  # (5, 2)

        # Stage 3: object mask
        print(f"  Segmenting '{object_name}'...")
        obj_mask = segment_object_gsam2(
            frame_rgb, object_name,
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
        )
        if obj_mask is None:
            print("  Object not found, skipping event.")
            continue

        # Filter fingertips on mask
        on_obj = np.array([
            (0 <= int(uv[1]) < H and 0 <= int(uv[0]) < W
             and obj_mask[int(uv[1]), int(uv[0])])
            for uv in ft2d
        ])
        contact_ft2d = ft2d[on_obj] if on_obj.any() else ft2d

        # Stage 4: depth + lift
        print("  Estimating depth...")
        depth_map = estimate_depth(frame_rgb, model_name=depth_model)
        if robot_depth is not None:
            depth_map = align_depth_scale(depth_map, robot_depth, obj_mask)

        contact_pts3d  = lift_to_3d(contact_ft2d, depth_map, camera_K)
        contact_point  = contact_pts3d.mean(axis=0)   # (3,)

        # Object centroid: lift all masked pixels to 3D, take mean.
        # Used to compute object-relative contact (camera-frame-independent).
        mask_pixels = np.column_stack(np.where(obj_mask))   # (N, 2) [row, col]
        mask_px2d   = mask_pixels[:, [1, 0]].astype(float)  # (N, 2) [col, row]
        if len(mask_px2d) > 500:
            idx = np.random.choice(len(mask_px2d), 500, replace=False)
            mask_px2d = mask_px2d[idx]
        obj_pts3d       = lift_to_3d(mask_px2d, depth_map, camera_K)   # (N, 3)
        obj_centroid    = obj_pts3d.mean(axis=0)                        # (3,)
        contact_relative = contact_point - obj_centroid                 # (3,)

        # Approach direction: mean wrist velocity in the 10 frames before grasp
        pre = [joint_sequence[i] for i in range(max(0, start_f-10), start_f)
               if joint_sequence[i] is not None]
        if len(pre) >= 2:
            wrist_traj = np.array([j[0] for j in pre])
            approach_dir = np.diff(wrist_traj, axis=0).mean(axis=0)
        else:
            approach_dir = np.array([0.0, 0.0, 1.0])
        norm = np.linalg.norm(approach_dir)
        approach_dir = approach_dir / norm if norm > 1e-6 else approach_dir

        # GraspGeom OBB: PCA on lifted object points → which face did the human approach?
        print("  Running GraspGeom PCA on object point cloud...")
        obb_axes, obb_sizes, face_offset_normalized = run_graspgeom_pca(
            obj_pts3d, approach_dir, obj_centroid
        )

        # Wrist pose (translation from joint 0; orientation TBD)
        wrist_pose = np.concatenate([joints_3d[0], np.zeros(3)], axis=0)  # (6,)

        # Stage 5: CLIP embedding
        try:
            obj_emb = compute_clip_embedding(frame_rgb, obj_mask)
        except Exception as e:
            print(f"  CLIP failed ({e}), skipping embedding.")
            obj_emb = None

        heatmap = _build_heatmap(contact_ft2d, H, W)

        prior = GraspPrior(
            contact_point_3d=contact_point,
            approach_direction=approach_dir,
            contact_point_relative=contact_relative,
            object_centroid_3d=obj_centroid,
            obb_axes=obb_axes,
            obb_sizes=obb_sizes,
            face_offset_normalized=face_offset_normalized,
            wrist_pose=wrist_pose,
            object_class=object_name,
            object_embedding=obj_emb,
            object_mask=obj_mask,
            contact_heatmap=heatmap,
            grasp_event=(int(start_f), int(end_f)),
        )
        print(f"  {prior.summary()}")
        priors.append(prior)

        if output_path is not None:
            prior.save(str(output_path).rstrip(".npz") + f"_{ev_idx}")

    print(f"\n[extract] Done — {len(priors)} prior(s) extracted.")
    return priors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",    required=True)
    ap.add_argument("--object",   required=True, help="e.g. 'mug'")
    ap.add_argument("--output",   required=True, help="e.g. priors/mug_demo")
    ap.add_argument("--no-hawor", action="store_true",
                    help="Use MediaPipe instead of HaWoR (no MANO needed)")
    ap.add_argument("--depth_model", default="depth-anything/Depth-Anything-V2-Small-hf")
    args = ap.parse_args()

    extract_prior_from_video(
        video_path=args.video,
        object_name=args.object,
        output_path=args.output,
        use_hawor=not args.no_hawor,
        depth_model=args.depth_model,
    )
