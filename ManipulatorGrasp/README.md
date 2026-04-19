# GraspNet + Human-Prior Grasp System with GraspGeom Reranking

A robot arm manipulation system that uses **human egocentric demonstration videos** to guide grasp selection on a UR5e arm with a Robotiq 2F-85 gripper in MuJoCo simulation.

The key contribution is the integration of **GraspGeom** into the GraspNet reranking pipeline: GraspGeom's OBB geometry analysis provides two additional scoring terms that, combined with human-prior contact and approach signals, produce stable and semantically meaningful grasp selection from GraspNet candidates.

---

## How GraspGeom Connects to GraspNet

GraspNet generates a large set of grasp candidates from a live RGB-D point cloud. On its own, GraspNet ranks candidates purely by grasp quality score — it has no knowledge of which part of the object the human intended to grasp, or how the object's geometry should constrain the approach.

This system bridges that gap in three stages:

### Stage 1 — Extract a human grasp prior from video

A single egocentric demo video is processed offline (in the `vitra` environment) to produce a `GraspPrior` `.npz` file containing:

- `contact_point_3d`: where the human's fingers landed on the object
- `approach_direction`: the wrist trajectory direction as the hand closed
- `obb_axes` / `obb_sizes`: OBB of the object in the demo camera frame (via GraspGeom's `estimate_object_axes`)
- `face_offset_normalized`: which OBB face was contacted, normalised by OBB half-sizes

### Stage 2 — Align the prior to the robot camera frame at execution time

`resolve_prior_to_robot_frame` in `main.py`:

1. Runs RANSAC plane removal on the live depth point cloud to isolate object points.
2. Calls GraspGeom's `estimate_object_axes` to fit a fresh OBB in the robot camera frame.
3. Maps `face_offset_normalized` through the robot OBB to get a 3D contact target.
4. Projects the demo approach direction through the OBB frame to get a camera-frame approach vector — this removes camera-specific tilt while preserving the human's intended grasp axis.

No explicit camera calibration is needed; the OBB provides a shared geometric reference between demo and robot.

### Stage 3 — Rerank GraspNet candidates with GraspGeom geometry terms

`rerank_with_graspgeom` in `main.py` scores all N GraspNet candidates on six terms:

| Term | Weight | Description |
|------|--------|-------------|
| GraspNet confidence | 0.15 | Normalised GraspNet quality score |
| Contact proximity | 0.25 | Gaussian distance to video contact point (σ = ½ OBB long axis) |
| Approach alignment | 0.20 | Cosine similarity to video approach direction |
| OBB axis alignment | 0.10 | Max `\|cos\|` of grasp approach vs any OBB principal axis |
| **GraspGeom face proximity** | **0.15** | Min Gaussian distance to OBB face centers from `estimate_contact_points` |
| **GraspGeom dominant axis** | **0.10** | `\|cos\|` alignment to dominant PCA axis from `analyze_grasp_geometry` |

The two GraspGeom terms (bold) are the new contribution. `estimate_contact_points` returns the four OBB face centers most reachable from the video approach direction — candidates whose gripper tip lands near one of these faces score highly. `analyze_grasp_geometry` identifies which OBB axis dominates the approach geometry; candidates aligned to that axis are preferred.

After reranking, the top candidate's **rotation matrix is replaced** with a stable OBB-derived orientation (approach vector + OBB closing axis with enforced sign consistency). This eliminates run-to-run gripper twist caused by sign ambiguity in GraspNet's output rotation matrices.

```
Human Egocentric Video
        |
        |-- HaWoR -----------------> right-hand MANO joints (21 x 3D, per frame)
        |-- Wrist speed minima -----> grasp event windows  (start_frame, end_frame)
        |-- Grounded-SAM2 ----------> object pixel mask (H x W)
        |-- Depth Anything V2 ------> depth map -> contact point 3D
        |-- CLIP -------------------> object embedding (512-d)
        |-- GraspGeom (estimate_object_axes) -> OBB axes + sizes in demo frame
        |
        v
   GraspPrior (.npz)
        |
        v
RGB-D Point Cloud (robot camera)
        |
        |-- GraspNet -----------------> N grasp candidates (position + rotation + score)
        |-- GraspGeom (estimate_object_axes) -> live OBB in robot camera frame
        |
        v
   resolve_prior_to_robot_frame
        |  face_offset_normalized × live OBB → robot contact point
        |  demo approach direction mapped through OBB frame
        |
        v
   rerank_with_graspgeom
        |  GraspNet score (0.15) + contact proximity (0.25) + approach alignment (0.20)
        |  + OBB axis alignment (0.10)
        |  + GraspGeom: estimate_contact_points → face proximity (0.15)
        |  + GraspGeom: analyze_grasp_geometry  → dominant axis (0.10)
        |
        v
   Top-ranked candidate → stable OBB rotation applied → execute_grasp → UR5e
```

---

## Key Files

| File | Role |
|------|------|
| `main.py` | Entry point; contains `rerank_with_graspgeom`, `detect_object_with_obb`, `resolve_prior_to_robot_frame`, `build_direct_grasp`, `execute_grasp` |
| `human_prior/reranker.py` | Base scoring helpers (`_contact_scores`, `_direction_scores`, `_obb_axis_scores`) used by `rerank_with_graspgeom` |
| `human_prior/prior.py` | `GraspPrior` dataclass — save/load `.npz` |
| `human_prior/extract.py` | Video → GraspPrior pipeline |
| `manipulator_grasp/env/ur5_grasp_env.py` | UR5 MuJoCo simulation environment |
| `manipulator_grasp/assets/scenes/scene.xml` | MuJoCo scene: UR5e + Robotiq 2F-85 + scissors |
| `graspnet-baseline/` | GraspNet model code (submodule) |
| `priors/` | GraspPrior `.npz` files extracted from demo videos |

### External dependencies (expected as sibling directories)

```
../GraspGeom/                  OBB-based grasp geometry analysis
../VITRA/thirdparty/HaWoR/     Hand reconstruction model
../Grounded-SAM-2/             Object segmentation
../Depth-Anything-V2/          Monocular depth estimation
```

---

## Conda Environments

| Environment | Python | Purpose |
|-------------|--------|---------|
| `graspnet`  | 3.9    | Robot execution: GraspNet + GraspGeom reranker + MuJoCo sim |
| `vitra`     | 3.10   | Video processing: HaWoR + SAM2 + Depth + CLIP |

---

## Setup

### 1. Clone and install

```bash
git clone <this-repo> Manipulator_Grasp
cd Manipulator_Grasp
git submodule update --init --recursive
```

### 2. Create the graspnet environment (Python 3.9)

```bash
conda create -n graspnet python=3.9
conda activate graspnet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/openai/CLIP.git"
pip install graspnetAPI open3d mujoco roboticstoolbox-python spatialmath-python
```

### 3. Create the vitra environment (Python 3.10)

```bash
conda create -n vitra python=3.10
conda activate vitra
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Grounded-SAM-2 (from sibling directory)
cd ../Grounded-SAM-2 && pip install -e . --no-build-isolation && cd -

pip install \
    mediapipe "git+https://github.com/openai/CLIP.git" \
    transformers supervision timm pycocotools \
    matplotlib scipy scikit-image huggingface_hub \
    natsort joblib easydict einops loguru yacs smplx \
    ultralytics "pytorch-lightning==2.2.4" \
    "torchmetrics==1.4.0" lightning-utilities \
    "pytorch-minimize" evo "setuptools<70"
```

### 4. Install GraspGeom

See the GraspGeom repository for setup instructions. It must be on the Python path before running either execution path. GraspGeom is used in both the prior extraction step (OBB fitting on the demo video depth map) and at robot execution time (live OBB + reranking geometry).

### 5. Model weights

| Model | Expected location |
|-------|-------------------|
| GraspNet checkpoint | `logs/log_rs/checkpoint-rs.tar` |
| HaWoR | `../VITRA/thirdparty/HaWoR/weights/hawor/checkpoints/hawor.ckpt` |
| HaWoR infiller | `../VITRA/thirdparty/HaWoR/weights/hawor/checkpoints/infiller.pt` |
| HaWoR detector | `../VITRA/thirdparty/HaWoR/weights/external/detector.pt` |
| SAM2 | `../Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt` |
| MANO right | `../VITRA/thirdparty/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl` |
| MANO left | `../VITRA/thirdparty/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl` |

Depth Anything V2 and Grounding DINO download automatically on first run via HuggingFace.

---

## Usage

### Step 1 — Record a human demonstration video

- Egocentric (first-person) video of your right hand grasping the target object.
- Phone camera, GoPro, or any RGB video — no depth required.

### Step 2 — Extract a grasp prior

```bash
conda activate vitra
python -m human_prior.extract \
    --video /path/to/demo.mp4 \
    --object "scissors" \
    --output priors/scissors_demo
```

| Argument | Description |
|----------|-------------|
| `--video` | Path to the demonstration video |
| `--object` | Text label for the target object |
| `--output` | Output path stem — saves `<stem>_0.npz`, `<stem>_1.npz`, etc. per grasp event |
| `--no-hawor` | Use MediaPipe instead of HaWoR (faster, less accurate) |

### Step 3 — Run GraspNet + GraspGeom reranking

```bash
conda activate graspnet
python main.py --prior priors/scissors_demo_0.npz --region handle
```

GraspNet generates candidates from the live depth image. `rerank_with_graspgeom` scores each candidate on the six terms above. The top-ranked candidate's position and rotation are both finalized using OBB geometry and executed on the UR5e.

Add `--direct` when GraspNet produces few candidates on thin objects (the reranker still runs but the final rotation is built entirely from OBB axes):

```bash
python main.py --prior priors/scissors_demo_0.npz --region handle --direct
```

---

## GraspPrior Format

| Field | Shape | Description |
|-------|-------|-------------|
| `contact_point_3d` | `(3,)` | Contact centroid in demo camera frame |
| `approach_direction` | `(3,)` | Unit vector: wrist trajectory toward object |
| `contact_point_relative` | `(3,)` | Contact offset from object centroid |
| `object_centroid_3d` | `(3,)` | Object centroid in demo camera frame |
| `obb_axes` | `(3,3)` | OBB axis matrix from GraspGeom (cols = long/mid/short) |
| `obb_sizes` | `(3,)` | OBB half-sizes in metres |
| `face_offset_normalized` | `(3,)` | Contact face offset normalised by OBB sizes |
| `wrist_pose` | `(6,)` | `[tx, ty, tz, rx, ry, rz]` at contact frame |
| `object_embedding` | `(512,)` | CLIP ViT-B/32 image embedding |
| `object_mask` | `(H,W)` bool | Object pixel mask |
| `contact_heatmap` | `(H,W)` float | Fingertip density map |
| `grasp_event` | `(2,)` int | `[start_frame, end_frame]` |

```python
from human_prior import GraspPrior
prior = GraspPrior.load("priors/scissors_demo_0.npz")
print(prior.contact_point_3d)       # (3,) float32
print(prior.approach_direction)     # (3,) float32, unit vector
print(prior.object_class)           # "scissors"
```

---

## Scissors Scene

The MuJoCo scene (`manipulator_grasp/assets/scenes/scene.xml`) contains a UR5e arm, Robotiq 2F-85 gripper, and scissors with capsule-frame handles.

| Parameter | Value |
|-----------|-------|
| Gripper pad friction | 40.0 / 35.0 |
| Table friction | 20.0 |
| Scissors total mass | ~0.5 g |
| Grasp width | 4 cm |
| Closing axis | short OBB axis (vertical), sign-stabilised |

