import os.path
import sys

sys.path.append('../../manipulator_grasp')

import time
import numpy as np
import spatialmath as sm
import mujoco
import mujoco.viewer

from manipulator_grasp.arm.robot import Robot, UR5e
from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.utils import mj


class UR5GraspEnv:

    def __init__(self):
        self.sim_hz = 500

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        self.robot_q = np.zeros(6)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()

        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None
        self.height = 256
        self.width = 256
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)
        self.num_points = 4096

    def reset(self):
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene.xml')
        self.mj_model = mujoco.MjModel.from_xml_path(filename)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.robot = UR5e()
        self.robot.set_base(mj.get_body_pose(self.mj_model, self.mj_data, "ur5e_base").t)
        self.robot_q = np.array([0.0, 0.0, np.pi / 2 * 0, 0.0, -np.pi / 2 * 0, 0.0])
        self.robot.set_joint(self.robot_q)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                            "wrist_2_joint", "wrist_3_joint"]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.robot_q[i]) for i, jn in enumerate(self.joint_names)]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mj.attach(self.mj_model, self.mj_data, "attach", "2f85", self.robot.fkine(self.robot_q))
        robot_tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        self.robot.set_tool(robot_tool)
        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()

        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.enable_depth_rendering()
        import time; time.sleep(1.0)   # let GLFW context fully settle before CUDA init
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        # Lock viewer to the fixed 'cam' camera (same view used for GraspNet inference)
        self.mj_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.mj_viewer.cam.fixedcamid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "cam"
        )

        self.camera_matrix = np.array([
            [self.height / (2.0 * np.tan(self.fovy / 2.0)), 0.0, self.width / 2.0],
            [0.0, self.height / (2.0 * np.tan(self.fovy / 2.0)), self.height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.step_num = 0
        # observation = self._get_obs()
        observation = None
        return observation

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()

    def start_recording(self, width: int = 640, height: int = 480,
                         fps: int = 30, every_n_steps: int = 2):
        """Set up frame capture. Call before the sim loop, then save_recording() at the end."""
        self._rec_renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
        cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        self._rec_renderer.update_scene(self.mj_data, cam_id)
        self._rec_frames = []
        self._rec_fps = fps
        self._rec_every = every_n_steps
        self._rec_step_count = 0

    def save_recording(self, path: str = "grasp_recording.mp4"):
        """Write captured frames to an MP4 file."""
        if not hasattr(self, '_rec_frames') or not self._rec_frames:
            print("[record] No frames captured.")
            return
        import imageio
        with imageio.get_writer(path, fps=self._rec_fps, format="ffmpeg",
                                codec="libx264", quality=8) as writer:
            for frame in self._rec_frames:
                writer.append_data(frame)
        print(f"[record] Saved {len(self._rec_frames)} frames → {path}")
        self._rec_renderer.close()

    def step(self, action=None):
        if action is not None:
            self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_viewer.sync()

        if hasattr(self, '_rec_frames'):
            self._rec_step_count += 1
            if self._rec_step_count % self._rec_every == 0:
                cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
                self._rec_renderer.update_scene(self.mj_data, cam_id)
                self._rec_frames.append(self._rec_renderer.render().copy())

    def render(self):
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render()
        }


if __name__ == '__main__':
    env = UR5GraspEnv()
    env.reset()
    for i in range(10000):
        env.step()
    imgs = env.render()
    env.close()
