# object_analysis

基于 **SAM3 文本分割 + MoGe2 单目深度 + PCA 姿态估计** 的物体抓取几何分析模块。  
输入一段操作视频和 VITRA 重建的手部状态，逐帧输出物体主轴、有向包围盒（OBB）及抓取接触点。

---

## 整体流程

```
视频帧
  │
  ├─→ [MoGe2]  单目几何估计 → 每像素深度 + 3D 点图 + 相机内参
  │
  ├─→ [SAM3]   文本概念分割（如 "scissors"）→ 物体二值 mask
  │
  ├─→ [手部状态] hand_state.npy → 腕部位置 + 接近方向 + 21 个关节 3D 坐标
  │
  ├─→ [点云提取] mask ∩ 深度有效区 − 手部区域 → 物体 3D 点云
  │
  ├─→ [PCA 姿态] 去离群点 → PCA 三主轴 + OBB 8 顶点
  │
  └─→ [抓取几何] 腕部→物体距离、接触角度、4 个候选接触点
```

---

## 模块说明

| 文件 | 功能 |
|---|---|
| `pipeline.py` | **主入口（CLI）**，串联所有步骤，逐帧处理并保存结果 |
| `segmentor.py` | `ObjectSegmentor`：封装 SAM3，文本提示 → `(H,W)` bool mask |
| `depth_estimator.py` | `DepthEstimator`：封装 MoGe2，含 NPZ 磁盘缓存，避免重复推理 |
| `cloud_extractor.py` | 从 SAM3 mask + MoGe2 点图提取物体点云，排除手部区域 |
| `pose_estimator.py` | PCA 估计三主轴（长/中/短）、尺寸和 OBB，返回 `ObjectAxes` |
| `grasp_analyzer.py` | 分析手腕–物体相对几何，返回 `GraspGeometry` 和候选接触点 |
| `hand_state.py` | 读取 `hand_state.npy`，提供腕部 transl、接近方向、简化 FK 关节位置 |
| `visualizer.py` | 在帧上叠加 mask、OBB、主轴箭头、接触点；可选 Open3D 三维视图 |

---

## 使用方法

```bash
cd F:/VLAproject/object_analysis

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
F:/Miniconda3/envs/vitra/python.exe -u pipeline.py \
    --video       F:/VLAproject/video/scissor_pick.mp4 \
    --hand_state  F:/VLAproject/data/hand_states/scissor_pick.hand_state.npy \
    --object_name "scissors" \
    --output_dir  F:/VLAproject/object_analysis/results \
    --save_video
```

> `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` 防止 MoGe2 启动时联网检查更新（模型已缓存时必须加）。

### 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--video` | 必填 | 输入视频路径 |
| `--hand_state` | 必填 | VITRA 输出的 `.hand_state.npy` |
| `--object_name` | 必填 | 物体名称（英文），作为 SAM3 文本提示 |
| `--output_dir` | `./results` | 输出目录 |
| `--stride` | `1` | 每隔几帧处理一次 |
| `--save_video` | 关 | 输出可视化 `.mp4` |
| `--depth_cache` | `<output_dir>/depths` | MoGe2 缓存目录 |
| `--sam_ckpt` | 自动 | SAM3 权重路径（默认 `../facebook/sam3/sam3.pt`） |
| `--min_points` | `30` | 点云最少有效点数 |

---

## 输出结构

```
results/
├── vis/
│   ├── frame_00010.jpg   # 每帧可视化（mask + OBB + 主轴 + 接触点）
│   └── ...
├── depths/
│   └── frame_00010.npz   # MoGe2 深度缓存，下次运行自动复用
├── results.json          # 每帧结果（可序列化：主轴、抓取几何、接触点）
├── results.npy           # 完整结果（含 3D 点云、手部关节，pickle）
└── analysis_vis.mp4      # 可视化视频（--save_video 时生成）
```

### `results.json` 单帧字段

```json
{
  "frame_id": 10,
  "object_name": "scissors",
  "obj_axes": {
    "center": [x, y, z],
    "long_axis": [dx, dy, dz],   // 第 1 主成分（最长方向）
    "mid_axis":  [dx, dy, dz],
    "short_axis": [dx, dy, dz],
    "long_size": 0.18,           // 单位同深度图（米）
    "mid_size": 0.06,
    "short_size": 0.03,
    "explained_variance_ratio": [0.82, 0.12, 0.06],
    "obb_vertices": [[...], ...]  // 8×3 OBB 顶点
  },
  "grasp_geo": {
    "wrist_pos": [...],
    "approach_dir": [...],
    "dist_wrist_obj": 0.15,
    "angle_long": 12.5,           // 接近方向与长轴夹角（度）
    "dominant_axis": "long"
  },
  "contact_pts": [[...], [...], [...], [...]],  // 4 个候选接触点
  "n_points": 1842
}
```

---

## 模型依赖

| 模型 | 用途 | 权重位置 |
|---|---|---|
| SAM3 | 文本概念分割 | `F:/VLAproject/facebook/sam3/sam3.pt` |
| MoGe2 (`moge-2-vitl`) | 单目几何估计（深度 + 点图） | HuggingFace 缓存（已离线） |


