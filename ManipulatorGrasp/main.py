import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import spatialmath as sm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GRASPNET_BASELINE = os.path.join(ROOT_DIR, 'graspnet-baseline')
if not os.path.isdir(os.path.join(GRASPNET_BASELINE, 'models')):
    GRASPNET_BASELINE = os.path.join(os.path.dirname(ROOT_DIR), 'graspnet-baseline')
sys.path.append(os.path.join(GRASPNET_BASELINE, 'models'))
sys.path.append(os.path.join(GRASPNET_BASELINE, 'dataset'))
sys.path.append(os.path.join(GRASPNET_BASELINE, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

# GraspNet and torch imported lazily inside get_net() / get_and_process_data()
# so that MuJoCo's OpenGL context is created before CUDA initialises.
from graspnetAPI import GraspGroup
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from collision_detector import ModelFreeCollisionDetector

from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

GRASPGEOM_ROOT = "/media/yutao/T91/projects/GraspGeom"
if GRASPGEOM_ROOT not in sys.path:
    sys.path.insert(0, GRASPGEOM_ROOT)

# Human-prior reranker (optional — only active when a prior .npz is provided)
from human_prior import GraspPrior, rerank_geometric, rerank_weighted, score_breakdown


def get_net():
    import torch
    from graspnet import GraspNet, pred_decode
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    checkpoint_path = './logs/log_rs/checkpoint-rs.tar'
    map_loc = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=map_loc)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


def get_and_process_data(imgs):
    num_point = 20000

    # imgs = np.load(os.path.join(data_dir, 'imgs.npz'))
    color = imgs['img'] / 255.0
    depth = imgs['depth']

    height = 256
    width = 256
    fovy = np.pi / 4
    intrinsic = np.array([
        [height / (2.0 * np.tan(fovy / 2.0)), 0.0, width / 2.0],
        [0.0, height / (2.0 * np.tan(fovy / 2.0)), height / 2.0],
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0

    camera = CameraInfo(height, width, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    mask = depth < 2.0
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    import torch
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud


def get_grasps(net, end_points):
    import torch
    from graspnet import pred_decode
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud):
    voxel_size = 0.01
    collision_thresh = 0.01

    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]

    return gg


def vis_grasps(gg, cloud):
    # gg.nms()
    # gg.sort_by_score()
    # gg = gg[:1]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


def detect_object_with_obb(cloud_points: np.ndarray):
    """RANSAC table removal + GraspGeom PCA on the remaining object points.

    Returns:
        centroid  : (3,) object centroid in robot camera frame.
        obj_axes  : ObjectAxes (GraspGeom) or None if PCA failed.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_points)

    _, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=100
    )
    obj_pcd = pcd.select_by_index(inliers, invert=True)

    if len(obj_pcd.points) == 0:
        return cloud_points.mean(axis=0), None

    obj_pts = np.asarray(obj_pcd.points, dtype=np.float32)
    centroid = obj_pts.mean(axis=0)

    try:
        from pose_estimator import estimate_object_axes
        obj_axes = estimate_object_axes(obj_pts)
    except Exception as e:
        print(f"[OBB] GraspGeom PCA failed ({e}), using centroid-only fallback.")
        obj_axes = None

    return centroid, obj_axes


def _find_handle_end_offset(cloud_points: np.ndarray, robot_centroid: np.ndarray,
                             robot_obb) -> np.ndarray:
    """Return face_offset_normalized pointing toward the handle end.

    Computes a fixed 20% long-axis offset toward the handle side.  This avoids
    noise from thin-object point clouds where the half-centroid approach is
    unreliable.  20% of long-axis length is a reasonable fraction for scissor
    handles (~33mm handle vs ~184mm OBB long-axis = 18%).

    Handle side = higher camera-X in this scene (world -Y → camera +X).
    """
    R = robot_obb.axis_matrix()    # cols = [long, mid, short]
    long_axis = R[:, 0]            # unit vector along long axis in camera frame

    # Decide handle sign: handle end → higher camera X
    handle_sign = +1.0 if long_axis[0] >= 0 else -1.0

    handle_cam_x = (robot_centroid + long_axis * handle_sign * 0.01)[0]
    other_cam_x  = (robot_centroid - long_axis * handle_sign * 0.01)[0]
    print(f"[handle detect] long_axis cam={long_axis.round(3)}, "
          f"handle side cam-X={handle_cam_x:.3f}  other cam-X={other_cam_x:.3f}")

    # Fixed 20% normalized offset toward handle end
    offset_norm = np.array([handle_sign * 0.20, 0.0, 0.0], dtype=np.float32)
    print(f"[handle detect] face_offset_normalized = {offset_norm}")
    return offset_norm


def resolve_prior_to_robot_frame(prior: "GraspPrior", cloud_points: np.ndarray,
                                  grasp_region: str = "auto") -> "GraspPrior":
    """Re-express the GraspPrior contact point and approach direction in the
    robot camera frame using the live point cloud.

    Args:
        prior        : loaded GraspPrior (video-derived).
        cloud_points : (N, 3) object point cloud in robot camera frame.
        grasp_region : "auto"   — use the video-derived face_offset_normalized.
                       "handle" — override contact to the handle end (bulkier
                                  long-axis end) while keeping video approach dir.
                       "blade"  — override contact to the blade/tip end.

    The returned prior has contact_point_3d and approach_direction set in the
    robot camera frame, ready for the reranker.
    """
    import dataclasses

    robot_centroid, robot_obb = detect_object_with_obb(cloud_points)

    # --- Override face_offset_normalized when a grasp_region is specified ---
    face_offset_override = None
    if grasp_region in ("handle", "blade") and robot_obb is not None:
        handle_offset = _find_handle_end_offset(cloud_points, robot_centroid, robot_obb)
        if grasp_region == "handle":
            face_offset_override = handle_offset
        else:  # "blade" = the opposite end
            face_offset_override = -handle_offset
        print(f"[align] Region override '{grasp_region}': "
              f"face_offset_norm = {face_offset_override}")

    # Effective face_offset: override wins; fall back to video prior
    face_offset_eff = (face_offset_override
                       if face_offset_override is not None
                       else prior.face_offset_normalized)

    # --- Primary path: OBB face alignment ---
    if face_offset_eff is not None and robot_obb is not None:
        R_robot = robot_obb.axis_matrix()  # (3, 3) cols=[long,mid,short], robot cam

        robot_obb_sizes = np.array([robot_obb.long_size, robot_obb.mid_size,
                                    robot_obb.short_size], dtype=np.float32)
        face_offset_metres = face_offset_eff * robot_obb_sizes

        # Reconstruct target face center in robot camera frame
        robot_contact = robot_centroid + R_robot @ face_offset_metres

        # --- Snap contact to the actual visible object surface ---
        # The point cloud only captures the top surface (depth camera looks down).
        # Filter to object points near the handle end (within mid_size in the
        # plane perpendicular to the long axis) and replace the contact with their
        # centroid.  This grounds the contact to the real depth-image surface rather
        # than relying on pure OBB geometry.
        if len(cloud_points) > 0:
            # Remove table points first
            _pcd_tmp = o3d.geometry.PointCloud()
            _pcd_tmp.points = o3d.utility.Vector3dVector(cloud_points)
            _, _inliers = _pcd_tmp.segment_plane(
                distance_threshold=0.01, ransac_n=3, num_iterations=100)
            _obj_pts = np.asarray(
                _pcd_tmp.select_by_index(_inliers, invert=True).points,
                dtype=np.float32)
            if len(_obj_pts) > 0:
                long_axis = R_robot[:, 0]
                long_proj = _obj_pts @ long_axis
                contact_long = robot_contact @ long_axis
                handle_half = float(robot_obb_sizes[0] * 0.35)
                near_handle = _obj_pts[
                    np.abs(long_proj - contact_long) < handle_half]
                if len(near_handle) > 5:
                    surface_contact = near_handle.mean(axis=0)
                    print(f"[align] Surface contact   : {surface_contact.round(3)} "
                          f"(from {len(near_handle)} pts)")
                    robot_contact = surface_contact

        # Approach direction: which OBB axis did the human approach along?
        # Project the demo wrist direction into OBB frame, keep only the dominant
        # axis — this removes camera-specific vertical/tilt contamination — then
        # map that single axis back to the robot camera frame.
        # This uses the video directly while staying camera-independent.
        if prior.obb_axes is not None:
            R_demo = prior.obb_axes.astype(np.float32)       # cols=[long,mid,short] demo cam
            approach_in_obj = R_demo.T @ prior.approach_direction  # (3,) in OBB frame
            # Map full approach direction through OBB frame — preserves the tilt
            # the human had in the video (not just the dominant horizontal axis).
            robot_approach = R_robot @ approach_in_obj
            robot_approach = robot_approach / (np.linalg.norm(robot_approach) + 1e-9)
            # Ensure approach points INTO the target face (anti-parallel to face_offset)
            if np.dot(robot_approach, face_offset_metres) > 0:
                robot_approach = -robot_approach
            mode_str = "video full approach (OBB frame) → robot cam"
        else:
            face_normal_obj = face_offset_metres / (np.linalg.norm(face_offset_metres) + 1e-9)
            robot_approach = -(R_robot @ face_normal_obj)
            robot_approach = robot_approach / (np.linalg.norm(robot_approach) + 1e-9)
            mode_str = "OBB face normal (no demo axes)"
        print(f"[align] Mode              : {mode_str}")

        print(f"[align] Robot obj centroid: {robot_centroid.round(3)}")
        print(f"[align] Robot OBB sizes   : {robot_obb_sizes.round(3)} m")
        print(f"[align] face_offset_norm  : {face_offset_eff.round(3)}")
        print(f"[align] face_offset_metres: {face_offset_metres.round(3)} m")
        print(f"[align] Robot contact pt  : {robot_contact.round(3)}")
        print(f"[align] Robot approach dir: {robot_approach.round(3)}")

        return dataclasses.replace(
            prior,
            contact_point_3d=robot_contact,
            approach_direction=robot_approach,
            robot_obb_axes=R_robot,
        )

    # --- Fallback: raw centroid offset ---
    if prior.contact_point_relative is not None:
        robot_contact = robot_centroid + prior.contact_point_relative

        print(f"[align] Mode              : centroid offset (no OBB)")
        print(f"[align] Robot obj centroid: {robot_centroid.round(3)}")
        print(f"[align] Contact offset    : {prior.contact_point_relative.round(3)}")
        print(f"[align] Robot contact pt  : {robot_contact.round(3)}")

        R_robot = robot_obb.axis_matrix() if robot_obb is not None else None
        return dataclasses.replace(
            prior,
            contact_point_3d=robot_contact,
            robot_obb_axes=R_robot,
        )

    # --- Nothing to resolve ---
    return prior


def rerank_with_graspgeom(gg, aligned_prior, obj_axes, sigma: float = 0.05):
    """Rerank GraspNet candidates combining human-prior terms with GraspGeom geometry.

    Scoring terms (all normalised, weights renormalised to sum=1):
      - GraspNet confidence          (w=0.15)
      - Prior contact proximity      (w=0.25) — Gaussian σ=sigma
      - Prior approach alignment     (w=0.20) — cosine to video approach dir
      - OBB axis alignment           (w=0.10) — max |cos| vs any OBB axis
      - GraspGeom contact candidates (w=0.15) — min Gaussian dist to OBB face centers
      - GraspGeom dominant axis      (w=0.10) — |cos| to dominant PCA axis
      - CLIP object similarity       (w=0.05) — dropped when unavailable

    Falls back to rerank_weighted if obj_axes is None or GraspGeom import fails.
    """
    if obj_axes is None:
        return rerank_weighted(gg, aligned_prior, sigma=sigma)

    try:
        from grasp_analyzer import analyze_grasp_geometry, estimate_contact_points
    except ImportError as e:
        print(f"[graspgeom] import failed ({e}), falling back to rerank_weighted")
        return rerank_weighted(gg, aligned_prior, sigma=sigma)

    try:
        approach_dir = aligned_prior.approach_direction.astype(np.float32)
        contact_pt   = np.array(aligned_prior.contact_point_3d, dtype=np.float32)

        # Proxy wrist pos: back off along approach direction from contact
        wrist_proxy = contact_pt - approach_dir * 0.15

        geom = analyze_grasp_geometry(obj_axes, wrist_proxy, approach_dir)
        contact_cands = estimate_contact_points(obj_axes, approach_dir, n_candidates=4)
        contact_cands = np.asarray(contact_cands, dtype=np.float32)   # (K, 3)

        axis_names = ["long", "mid", "short"]
        print(f"[graspgeom] dominant={axis_names[geom.dominant_axis_idx]} "
              f"(comp={geom.dominant_axis_comp:.2f}), "
              f"angle_long={geom.angle_vs_long:.1f}°, "
              f"{len(contact_cands)} face candidates")

        # Dominant OBB axis vector in robot camera frame
        dominant_axis = obj_axes.axis_matrix()[:, geom.dominant_axis_idx].astype(np.float32)

    except Exception as e:
        print(f"[graspgeom] geometry analysis failed ({e}), falling back to rerank_weighted")
        return rerank_weighted(gg, aligned_prior, sigma=sigma)

    from human_prior.reranker import (_contact_scores, _direction_scores, _obb_axis_scores)

    translations  = gg.translations                    # (N, 3)
    approach_axes = gg.rotation_matrices[:, :, 1]      # (N, 3) col 1 = approach

    # --- GraspNet confidence (normalised) ---
    gn_scores = gg.scores.copy().astype(np.float32)
    gn_max = gn_scores.max()
    if gn_max > 1e-8:
        gn_scores /= gn_max

    # --- Prior contact proximity ---
    cs = _contact_scores(translations, aligned_prior.contact_point_3d, sigma)

    # --- Prior approach alignment ---
    ds = _direction_scores(approach_axes, aligned_prior.approach_direction)

    # --- OBB axis alignment ---
    w_obb = 0.10
    if aligned_prior.robot_obb_axes is not None:
        obb_scores = _obb_axis_scores(approach_axes, aligned_prior.robot_obb_axes)
    else:
        obb_scores = np.zeros(len(gg), dtype=np.float32)
        w_obb = 0.0

    # --- GraspGeom: contact candidate proximity ---
    dists_to_cands = np.linalg.norm(
        translations[:, None, :] - contact_cands[None, :, :], axis=2
    )   # (N, K)
    min_dists = dists_to_cands.min(axis=1)             # (N,)
    geom_contact_scores = np.exp(-min_dists ** 2 / (2.0 * sigma ** 2))

    # --- GraspGeom: dominant axis alignment (|cos|, mapped to [0,1]) ---
    cos_dom = approach_axes @ dominant_axis             # (N,)
    geom_axis_scores = np.abs(np.clip(cos_dom, -1.0, 1.0))

    # --- Weights ---
    w_graspnet    = 0.15
    w_contact     = 0.25
    w_direction   = 0.20
    w_object      = 0.00   # CLIP: no scene embedding here; zero weight
    w_geom_contact = 0.15
    w_geom_axis   = 0.10

    total_w = (w_graspnet + w_contact + w_direction + w_obb
               + w_geom_contact + w_geom_axis + w_object)
    w_graspnet    /= total_w
    w_contact     /= total_w
    w_direction   /= total_w
    w_obb         /= total_w
    w_geom_contact /= total_w
    w_geom_axis   /= total_w

    combined = (w_graspnet     * gn_scores
              + w_contact      * cs
              + w_direction    * ds
              + w_obb          * obb_scores
              + w_geom_contact * geom_contact_scores
              + w_geom_axis    * geom_axis_scores)

    order    = np.argsort(-combined)
    reranked = gg[order]
    reranked._combined_scores = combined[order]
    return reranked


def generate_grasps(net, imgs, visual=False, num_grasps=1, prior=None,
                     grasp_region: str = "auto", position_override: bool = False):
    """Generate and optionally rerank GraspNet candidates using a human prior.

    Args:
        net               : GraspNet model.
        imgs              : dict with 'img' (H,W,3) and 'depth' (H,W).
        visual            : open Open3D viewer if True.
        num_grasps        : number of top grasps to return.
        prior             : GraspPrior loaded from a .npz file, or None.
        grasp_region      : "auto" / "handle" / "blade" — which face to target.
        position_override : If True and prior is set, keep the reranker's best
                            rotation matrix but replace the translation with the
                            prior contact point (handle/blade center from OBB).
                            Use when GraspNet candidates are on wrong part of object.

    Returns:
        gg_final    : GraspGroup (top num_grasps), reranked if prior given else GraspNet-sorted.
        gg_original : GraspGroup (top num_grasps), always GraspNet-score sorted.
                      Identical to gg_final when no prior is provided.
    """
    end_points, cloud = get_and_process_data(imgs)
    gg = get_grasps(net, end_points)
    cloud_points = np.array(cloud.points)
    gg = collision_detection(gg, cloud_points)

    # Always build the original GraspNet ranking for comparison
    gg.nms()
    gg.sort_by_score()
    gg_original = gg[:num_grasps]

    if prior is not None and len(gg) > 0:
        aligned_prior = resolve_prior_to_robot_frame(prior, cloud_points,
                                                      grasp_region=grasp_region)

        print(f"\n[reranker] {len(gg)} candidates after collision filter "
              f"(object: {aligned_prior.object_class})")

        # --- Original GraspNet top result ---
        t0 = gg.translations[0].round(3)
        r0 = gg.scores[0]
        print(f"\n  GraspNet top-1 (before rerank):")
        print(f"    translation : {t0}")
        print(f"    score       : {r0:.4f}")

        # --- Sigma: half the robot OBB long axis (metres) for contact proximity ---
        _, _robot_obb = detect_object_with_obb(cloud_points)
        sigma = float(_robot_obb.long_size / 2.0) if _robot_obb is not None else 0.08

        # --- Score breakdown for top-5 ---
        print(f"\n  Score breakdown (top 5 candidates, sigma={sigma:.3f}m):")
        score_breakdown(gg, aligned_prior, top_k=5, sigma=sigma)

        # --- Rerank with GraspGeom geometry terms ---
        gg_reranked = rerank_with_graspgeom(gg, aligned_prior, _robot_obb, sigma)
        t1 = gg_reranked.translations[0].round(3)
        print(f"\n  Reranked top-1 (human-prior guided):")
        print(f"    translation : {t1}")
        print(f"    combined    : {gg_reranked._combined_scores[0]:.4f}")

        changed = not np.allclose(t0, t1, atol=1e-3)
        print(f"\n  Selection changed: {changed}")
        if changed:
            print(f"    Δ translation : {(t1 - t0).round(3)}")

        gg_final = gg_reranked[:num_grasps]

        # --- Position + rotation override: stable OBB-derived pose, reranker-selected contact ---
        if position_override and aligned_prior.contact_point_3d is not None:
            old_t = gg_final.translations[0].copy()
            contact_override = aligned_prior.contact_point_3d.copy().astype(np.float64)
            contact_override[0] += 0.030   # right in camera view
            contact_override[1] -= 0.033   # toward camera
            contact_override[2] += 0.020   # toward table plane (+cam_z = downward in world)
            gg_final.translations[0] = contact_override
            print(f"\n  [position override] translation {old_t.round(3)} "
                  f"→ hard-coded contact {contact_override.round(3)}")

            # Build a stable rotation from OBB geometry — prevents gripper twist from
            # run-to-run variation in GraspNet candidate orientations.
            if aligned_prior.robot_obb_axes is not None:
                approach = aligned_prior.approach_direction.copy().astype(np.float64)
                approach[2] *= 0.3
                approach /= np.linalg.norm(approach) + 1e-9

                closing = aligned_prior.robot_obb_axes[:, 2].copy().astype(np.float64)
                if closing[2] > 0:   # force consistent half-space — prevents 180° flip
                    closing = -closing
                closing = closing - (closing @ approach) * approach
                norm = np.linalg.norm(closing)
                if norm > 1e-6:
                    closing = closing / norm
                else:
                    closing = aligned_prior.robot_obb_axes[:, 1].astype(np.float64)
                    closing = closing - (closing @ approach) * approach
                    closing /= np.linalg.norm(closing) + 1e-9

                depth = np.cross(approach, closing)
                depth /= np.linalg.norm(depth) + 1e-9
                R_stable = np.stack([approach, closing, depth], axis=1)
                gg_final.rotation_matrices[0] = R_stable
                print(f"  [rotation override] OBB-derived stable rotation applied")
                print(f"    approach (cam): {approach.round(3)}")
                print(f"    closing  (cam): {closing.round(3)}")
    else:
        gg_final = gg_original

    if visual:
        print("\n[vis] Showing reranked grasp (green) alongside GraspNet top (all):")
        vis_grasps(gg_final, cloud)

    return gg_final, gg_original


def _safe_step(env, action):
    """Call env.step only if action contains no NaN/Inf. Returns True on success."""
    if not np.all(np.isfinite(action)):
        return False
    env.step(action)
    return True


def build_direct_grasp(prior: "GraspPrior", cloud_points: np.ndarray,
                        grasp_region: str = "handle") -> "GraspGroup":
    """Build a single synthetic grasp from OBB position + video orientation.

    Used when GraspNet fails to generate candidates on thin/small objects.
    The position comes from the OBB handle/blade end; the orientation is
    constructed from the video approach direction and the OBB closing axis.

    Args:
        prior        : GraspPrior already resolved to robot camera frame
                       (contact_point_3d and approach_direction set).
        cloud_points : (N,3) point cloud in robot camera frame.
        grasp_region : "handle" or "blade" — which end was targeted.

    Returns:
        GraspGroup with one synthetic grasp entry.
    """
    from graspnetAPI import GraspGroup

    contact = np.array(prior.contact_point_3d, dtype=np.float64)
    contact[0] += 0.030   # shift right in camera view (+cam_x = world -Y)
    contact[1] -= 0.033   # shift toward camera / user (-cam_y = world -X)
    contact[2] += 0.020   # shift toward table plane (+cam_z = downward in world)
    approach = prior.approach_direction.copy()         # (3,) unit vec, camera frame
    approach[2] *= 0.3    # flatten approach toward horizontal (reduce depth/above tilt)
    approach /= np.linalg.norm(approach) + 1e-9

    # Closing axis: OBB mid-axis (handle width direction = horizontal on table).
    # Scissors lie flat; their thickness (short axis) is only ~8 mm — gripping
    # vertically (across the thickness) would require the lower finger to go below
    # the table surface.  The mid-axis (~6 cm handle width) is horizontal, so
    # both fingers remain above the table and straddle the handle from the sides.
    if prior.robot_obb_axes is not None:
        # Short axis = col 2 = vertical (world Z); scissors elevated on stand so
        # both fingers remain above the table when closing vertically.
        closing = prior.robot_obb_axes[:, 2].copy()
        # Force consistent sign: cam_z points into the scene (downward in world).
        # Ensure closing[2] < 0 so the axis always points in the same half-space,
        # preventing 180° gripper flips across runs.
        if closing[2] > 0:
            closing = -closing
        closing = closing - (closing @ approach) * approach   # orthogonalise vs approach
        norm = np.linalg.norm(closing)
        if norm > 1e-6:
            closing = closing / norm
        else:
            closing = prior.robot_obb_axes[:, 1]              # fallback: mid-axis
            closing = closing - (closing @ approach) * approach
            closing /= np.linalg.norm(closing) + 1e-9
    else:
        # No OBB: use camera horizontal (cam X) as closing
        cam_horiz = np.array([1.0, 0.0, 0.0])
        closing = cam_horiz - (cam_horiz @ approach) * approach
        norm = np.linalg.norm(closing)
        closing = closing / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])

    # Col 0 = approach, Col 1 = closing (vertical), Col 2 = depth (right-hand rule).
    depth = np.cross(approach, closing)
    depth /= np.linalg.norm(depth) + 1e-9

    R = np.stack([approach, closing, depth], axis=1)  # cols=[approach, closing, depth]

    # GraspGroup row: [score, width, height, depth, R(9), t(3), obj_id]
    grasp_score = 0.9
    grasp_width = 0.04    # 4 cm — straddles 1.6 cm handle height with scissors on stand
    grasp_height = 0.02
    grasp_depth = 0.02
    obj_id = 0

    row = np.array([grasp_score, grasp_width, grasp_height, grasp_depth,
                    *R.flatten(), *contact, obj_id], dtype=np.float32)
    gg = GraspGroup(row[np.newaxis, :])

    print(f"[direct grasp] position  (cam) : {contact.round(3)}")
    print(f"[direct grasp] approach  (cam) : {approach.round(3)}  [col0 → end-on from handle tip]")
    print(f"[direct grasp] closing   (cam) : {closing.round(3)}   [col1 → fingers across handle width]")
    return gg


def execute_grasp(env, robot, gg, q0):
    n_wc = np.array([0.0, -1.0, 0.0])
    o_wc = np.array([-1.0, 0.0, -0.5])
    t_wc = np.array([1.0, 0.6, 2.0])
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(
        sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))

    T_wo = T_wc * T_co

    # Pre-check: reject grasps clearly outside the UR5e workspace (~0.85 m reach)
    robot_base = np.array(robot.base.t).flatten()
    dist_to_grasp = float(np.linalg.norm(T_wo.t - robot_base))
    if dist_to_grasp > 0.95:
        print(f"[execute_grasp] grasp world pos {np.array(T_wo.t).round(3)} is "
              f"{dist_to_grasp:.3f} m from base — out of workspace, skipping.")
        return False

    time0 = 2
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])

    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time0)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner0 = TrajectoryPlanner(trajectory_parameter0)

    time1 = 2
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)
    position_parameter1 = LinePositionParameter(T1.t, T2.t)
    attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R))
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
    velocity_parameter1 = QuinticVelocityParameter(time1)
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
    planner1 = TrajectoryPlanner(trajectory_parameter1)

    time2 = 2
    T3 = T_wo
    position_parameter2 = LinePositionParameter(T2.t, T3.t)
    attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time2)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    planner2 = TrajectoryPlanner(trajectory_parameter2)

    time_array = [0, time0, time1, time2]
    planner_array = [planner0, planner1, planner2]
    total_time = np.sum(time_array)

    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0, total_time, time_step_num)

    time_cumsum = np.cumsum(time_array)
    action = np.zeros(7)
    joint = q0.copy()
    for i, timei in enumerate(times):
        for j in range(len(time_cumsum)):
            if timei < time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    if robot.move_cartesian(planner_interpolate):
                        joint = robot.get_joint()
                action[:6] = joint
                _safe_step(env, action)
                break

    for i in range(1500):
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        _safe_step(env, action)

    time3 = 2
    T4 = sm.SE3.Trans(0.0, 0.0, 0.1) * T3
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time3)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner3 = TrajectoryPlanner(trajectory_parameter3)

    time4 = 2
    T5 = sm.SE3.Trans(1.4, 0.2, T4.t[2]) * sm.SE3(sm.SO3(T4.R))
    position_parameter4 = LinePositionParameter(T4.t, T5.t)
    attitude_parameter4 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
    velocity_parameter4 = QuinticVelocityParameter(time4)
    trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
    planner4 = TrajectoryPlanner(trajectory_parameter4)

    time5 = 2
    T6 = sm.SE3.Trans(0.2, 0.2, T5.t[2]) * sm.SE3(sm.SO3.Rz(-np.pi / 2) * sm.SO3(T5.R))
    position_parameter5 = LinePositionParameter(T5.t, T6.t)
    attitude_parameter5 = OneAttitudeParameter(sm.SO3(T5.R), sm.SO3(T6.R))
    cartesian_parameter5 = CartesianParameter(position_parameter5, attitude_parameter5)
    velocity_parameter5 = QuinticVelocityParameter(time5)
    trajectory_parameter5 = TrajectoryParameter(cartesian_parameter5, velocity_parameter5)
    planner5 = TrajectoryPlanner(trajectory_parameter5)

    time6 = 2
    T7 = sm.SE3.Trans(0.0, 0.0, -0.1) * T6
    position_parameter6 = LinePositionParameter(T6.t, T7.t)
    attitude_parameter6 = OneAttitudeParameter(sm.SO3(T6.R), sm.SO3(T7.R))
    cartesian_parameter6 = CartesianParameter(position_parameter6, attitude_parameter6)
    velocity_parameter6 = QuinticVelocityParameter(time6)
    trajectory_parameter6 = TrajectoryParameter(cartesian_parameter6, velocity_parameter6)
    planner6 = TrajectoryPlanner(trajectory_parameter6)

    time_array = [0.0, time3, time4, time5, time6]
    planner_array = [planner3, planner4, planner5, planner6]
    total_time = np.sum(time_array)

    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)

    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    if robot.move_cartesian(planner_interpolate):
                        joint = robot.get_joint()
                action[:6] = joint
                _safe_step(env, action)
                break

    for i in range(1500):
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        _safe_step(env, action)

    time7 = 2
    T8 = sm.SE3.Trans(0.0, 0.0, 0.2) * T7
    position_parameter7 = LinePositionParameter(T7.t, T8.t)
    attitude_parameter7 = OneAttitudeParameter(sm.SO3(T7.R), sm.SO3(T8.R))
    cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)

    time_array = [0.0, time7]
    planner_array = [planner7]
    total_time = np.sum(time_array)

    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)

    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    if robot.move_cartesian(planner_interpolate):
                        joint = robot.get_joint()
                action[:6] = joint
                _safe_step(env, action)
                break

    time8 = 2.0
    q8 = robot.get_joint()
    q9 = q0

    parameter8 = JointParameter(q8, q9)
    velocity_parameter8 = QuinticVelocityParameter(time8)
    trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
    planner8 = TrajectoryPlanner(trajectory_parameter8)

    time_array = [0.0, time8]
    planner_array = [planner8]
    total_time = np.sum(time_array)

    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)

    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    if robot.move_cartesian(planner_interpolate):
                        joint = robot.get_joint()
                action[:6] = joint
                _safe_step(env, action)
                break

    # settle after placing
    for i in range(500):
        env.step()

    return True


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=None,
                    help="Path to a GraspPrior .npz file extracted from a human demo video. "
                         "If provided, GraspNet candidates are reranked using the human prior. "
                         "Example: --prior priors/mug_demo_0.npz")
    ap.add_argument("--region", default="auto", choices=["auto", "handle", "blade"],
                    help="Grasp region on the object.  "
                         "'auto' uses the video-derived contact point.  "
                         "'handle' overrides to the bulkier long-axis end (e.g. scissors handles) "
                         "while keeping the video approach direction.  "
                         "'blade' targets the opposite (tip) end.  "
                         "Future: this will be set automatically by Qwen object detection.")
    ap.add_argument("--direct", action="store_true",
                    help="Direct grasp mode: skip GraspNet candidate selection. "
                         "Position is taken from OBB handle/blade detection; "
                         "orientation is built from the video approach direction + OBB closing axis. "
                         "Use when GraspNet fails to generate candidates on thin objects. "
                         "Requires --prior and --region handle|blade.")
    args = ap.parse_args()

    # Load human prior if provided
    prior = None
    if args.prior is not None:
        prior = GraspPrior.load(args.prior)
        print(f"\n{prior.summary()}\n")
        if args.region != "auto":
            print(f"[region override] Targeting '{args.region}' — "
                  f"video approach direction preserved, contact face overridden at runtime.\n")
    if args.direct and prior is None:
        print("ERROR: --direct requires --prior")
        import sys; sys.exit(1)

    # Pre-init CUDA before GLFW viewer so both contexts can coexist on the GPU
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.init()
    del _torch

    # Init MuJoCo env (creates GLFW viewer) after CUDA context is established
    env = UR5GraspEnv()
    env.reset()

    net = get_net()
    for i in range(1000):
        env.step()

    robot = env.robot
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for grasp_iter in range(1):  # single demo; increase for retry attempts
        print(f"\n--- Grasp attempt {grasp_iter + 1} ---")
        imgs = env.render()

        if args.direct:
            # Direct mode: GraspNet + reranker selects best candidate, then
            # position AND rotation are overridden with stable OBB geometry.
            region = args.region if args.region != "auto" else "handle"
            gg, gg_original = generate_grasps(net, imgs, visual=False, prior=prior,
                                               grasp_region=region,
                                               position_override=True)
        else:
            gg, gg_original = generate_grasps(net, imgs, visual=False, prior=prior,
                                               grasp_region=args.region)

        if len(gg) == 0:
            print("No grasps found, stopping.")
            break
        # gg_original holds the raw GraspNet top result (for analysis / ablation)
        success = execute_grasp(env, robot, gg, q0)
        if success is False:
            print(f"  Grasp attempt {grasp_iter + 1} skipped (out of workspace).")

    for i in range(2000):
        env.step()

    env.close()
