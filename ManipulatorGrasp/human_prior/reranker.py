"""
Grasp reranker — scores and reorders GraspNet candidates using a human prior.

Phases:
  Phase 1 — rerank_geometric()
    Pure geometry: contact proximity + approach alignment.

  Phase 2 — rerank_weighted()
    Weighted heuristic with four terms:
      - GraspNet confidence
      - Contact proximity (Gaussian)
      - Approach alignment (cosine to OBB face normal or wrist trajectory)
      - OBB axis alignment (rewards axis-aligned grasps regardless of which face)
      - CLIP object similarity (optional)

GraspNet rotation_matrix column convention:
  col 0  →  closing / width axis
  col 1  →  approach axis  (gripper moves toward object along this axis)
  col 2  →  depth axis
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .prior import GraspPrior

# graspnetAPI is only available in the graspnet conda env (Python 3.9).
# It is intentionally NOT imported at module level so that extract.py can
# import GraspPrior from this package while running in the vitra env (Python 3.10).
# Each function that needs GraspGroup imports it locally.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _approach_axes(gg: GraspGroup) -> np.ndarray:
    """Return (N, 3) approach axes from GraspGroup rotation matrices."""
    return gg.rotation_matrices[:, :, 1]   # col 1 = approach axis


def _contact_scores(translations: np.ndarray, contact_point: np.ndarray,
                    sigma: float) -> np.ndarray:
    """Gaussian proximity score: 1 when grasp centre == contact point."""
    dists = np.linalg.norm(translations - contact_point[None, :], axis=1)
    return np.exp(-dists ** 2 / (2.0 * sigma ** 2))


def _direction_scores(approach_axes: np.ndarray,
                      approach_direction: np.ndarray) -> np.ndarray:
    """Cosine similarity of grasp approach axis vs prior approach direction.
    Mapped to [0, 1]: 1 = perfectly aligned, 0 = anti-aligned."""
    cos = approach_axes @ approach_direction
    return (1.0 + np.clip(cos, -1.0, 1.0)) / 2.0


def _obb_axis_scores(approach_axes: np.ndarray,
                     robot_obb_axes: np.ndarray) -> np.ndarray:
    """Score by alignment of the grasp approach with any OBB principal axis.

    A GraspNet candidate that approaches along the object's long, mid, or short
    axis is preferred over one that approaches at an arbitrary angle.  We take
    the maximum absolute cosine across all three axes so that axis-aligned
    grasps score close to 1 regardless of which axis they align with.

    Args:
        approach_axes  : (N, 3) GraspNet approach directions.
        robot_obb_axes : (3, 3) OBB axis matrix in robot cam frame.
                         Columns are [long_axis, mid_axis, short_axis].
    Returns:
        (N,) scores in [0, 1].
    """
    # |cos| with each of the 3 OBB axes → take max
    dots = np.abs(approach_axes @ robot_obb_axes)   # (N, 3)
    return dots.max(axis=1)                          # (N,)


def _clip_sim(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine similarity between two CLIP embeddings."""
    a = emb_a / (np.linalg.norm(emb_a) + 1e-8)
    b = emb_b / (np.linalg.norm(emb_b) + 1e-8)
    return float(a @ b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank_geometric(
    gg,
    prior: GraspPrior,
    w_contact: float = 0.6,
    w_direction: float = 0.4,
    sigma: float = 0.05,
) -> GraspGroup:
    """Phase 1: rerank purely by geometric alignment to the human prior.

    Args:
        gg          : GraspNet candidates (already NMS + collision filtered).
        prior       : Human prior with contact_point_3d and approach_direction
                      already resolved to the robot camera frame.
        w_contact   : Weight for contact-point proximity.
        w_direction : Weight for approach-direction alignment.
        sigma       : Gaussian bandwidth in metres (~5 cm default).

    Returns:
        GraspGroup sorted by descending geometric score.
    """
    translations = gg.translations
    axes = _approach_axes(gg)

    cs = _contact_scores(translations, prior.contact_point_3d, sigma)
    ds = _direction_scores(axes, prior.approach_direction)

    scores = w_contact * cs + w_direction * ds
    order  = np.argsort(-scores)
    reranked = gg[order]
    reranked._geometric_scores = scores[order]
    return reranked


def rerank_weighted(
    gg,
    prior: GraspPrior,
    w_graspnet: float = 0.20,
    w_contact: float = 0.40,
    w_direction: float = 0.25,
    w_obb: float = 0.10,
    w_object: float = 0.05,
    sigma: float = 0.05,
    scene_object_embedding: Optional[np.ndarray] = None,
) -> GraspGroup:
    """Phase 2: weighted heuristic reranker with OBB axis alignment term.

    score(g) = w_graspnet  × GraspNet_confidence(g)     (normalised)
             + w_contact   × contact_proximity(g)        (Gaussian σ=5 cm)
             + w_direction × approach_alignment(g)       (cosine to OBB face normal)
             + w_obb       × obb_axis_alignment(g)       (max |cos| vs any OBB axis)
             + w_object    × CLIP_similarity(g)          (optional)

    The w_obb term rewards grasps that approach along one of the object's
    principal axes, even if they don't match the exact face.  It uses
    prior.robot_obb_axes, which is set by resolve_prior_to_robot_frame().
    If robot_obb_axes is not available, the w_obb weight is dropped and
    weights are renormalised automatically.

    Args:
        gg                     : GraspNet candidates.
        prior                  : Human prior resolved to robot camera frame.
        w_*                    : Term weights (renormalised if they don't sum to 1).
        sigma                  : Gaussian bandwidth for contact proximity.
        scene_object_embedding : CLIP embedding of the target in the robot scene.
                                 If None, the object term is dropped.

    Returns:
        GraspGroup sorted by descending combined score.
    """
    translations = gg.translations
    axes = _approach_axes(gg)

    # --- GraspNet confidence ---
    gn_scores = gg.scores.copy().astype(np.float32)
    max_gn = gn_scores.max()
    if max_gn > 1e-8:
        gn_scores /= max_gn

    # --- contact proximity ---
    cs = _contact_scores(translations, prior.contact_point_3d, sigma)

    # --- approach alignment (OBB face normal or wrist velocity) ---
    ds = _direction_scores(axes, prior.approach_direction)

    # --- OBB axis alignment ---
    if prior.robot_obb_axes is not None:
        obb_scores = _obb_axis_scores(axes, prior.robot_obb_axes)
    else:
        obb_scores = np.zeros(len(gg), dtype=np.float32)
        w_obb = 0.0

    # --- CLIP object similarity ---
    if prior.object_embedding is not None and scene_object_embedding is not None:
        sim = _clip_sim(prior.object_embedding, scene_object_embedding)
        obj_scores = np.full(len(gg), (1.0 + sim) / 2.0, dtype=np.float32)
    else:
        obj_scores = np.zeros(len(gg), dtype=np.float32)
        w_object = 0.0

    # --- normalise weights ---
    total_w = w_graspnet + w_contact + w_direction + w_obb + w_object
    if total_w < 1e-8:
        raise ValueError("All reranker weights are zero.")
    w_graspnet  /= total_w
    w_contact   /= total_w
    w_direction /= total_w
    w_obb       /= total_w
    w_object    /= total_w

    combined = (w_graspnet  * gn_scores
              + w_contact   * cs
              + w_direction * ds
              + w_obb       * obb_scores
              + w_object    * obj_scores)

    order = np.argsort(-combined)
    reranked = gg[order]
    reranked._combined_scores = combined[order]
    return reranked


def score_breakdown(
    gg,
    prior: GraspPrior,
    sigma: float = 0.05,
    top_k: int = 5,
) -> None:
    """Print a score breakdown for the top-k candidates (debugging)."""
    translations = gg.translations
    axes = _approach_axes(gg)
    cs   = _contact_scores(translations, prior.contact_point_3d, sigma)
    ds   = _direction_scores(axes, prior.approach_direction)
    gn   = gg.scores / (gg.scores.max() + 1e-8)

    has_obb = prior.robot_obb_axes is not None
    if has_obb:
        obb = _obb_axis_scores(axes, prior.robot_obb_axes)

    header = (f"\n{'#':>3}  {'GraspNet':>9}  {'Contact':>9}  {'Direction':>9}"
              + (f"  {'OBB-Axis':>9}" if has_obb else "")
              + f"  {'Translation':>30}")
    print(header)
    print("-" * (70 + (12 if has_obb else 0)))

    for i in range(min(top_k, len(gg))):
        t = translations[i]
        row = (f"{i:>3}  {gn[i]:>9.3f}  {cs[i]:>9.3f}  {ds[i]:>9.3f}"
               + (f"  {obb[i]:>9.3f}" if has_obb else "")
               + f"  [{t[0]:6.3f} {t[1]:6.3f} {t[2]:6.3f}]")
        print(row)
