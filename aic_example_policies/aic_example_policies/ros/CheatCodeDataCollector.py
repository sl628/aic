#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Ground-truth data collector built on top of CheatCode.

Run with ``ground_truth:=true`` so that the TF frames used by CheatCode
are available. Every insert_cable() invocation becomes one episode saved
to disk in a format that can be loaded directly for training.

Data layout on disk::

    <output_dir>/
      dataset_index.jsonl        # one line per episode (global index)
      episodes/
        <episode_id>/
          meta.json              # episode metadata
          data.parquet           # per-step state + action (tabular)
          images/
            left_camera/
              000000.jpg  000001.jpg  ...
            center_camera/
              ...
            right_camera/
              ...

State vector (26D, matching RunACT.py layout):
  [0:3]   tcp_pose position  (x, y, z)
  [3:7]   tcp_pose orientation (qx, qy, qz, qw)
  [7:10]  tcp_velocity linear  (x, y, z)
  [10:13] tcp_velocity angular (x, y, z)
  [13:19] tcp_error  (x, y, z, rx, ry, rz)
  [19:26] joint_positions  (7 values)

Action vector (7D, position-based):
  [0:3]   target TCP position    (x, y, z)  in base_link frame
  [3:7]   target TCP orientation (qx, qy, qz, qw)

Usage::

    pixi run ros2 run aic_model aic_model --ros-args \\
        -p use_sim_time:=true \\
        -p policy:=aic_example_policies.ros.CheatCodeDataCollector \\
        -p output_dir:=/path/to/output

You can also set the environment variable ``AIC_DATA_DIR`` as a fallback.
"""

import json
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from aic_control_interfaces.msg import MotionUpdate
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from rclpy.time import Time

from .CheatCode import CheatCode

_CAMERA_ROS_ATTRS = ["left_image", "center_image", "right_image"]
_CAMERA_KEYS = ["left_camera", "center_camera", "right_camera"]
_IMAGE_SCALE = 0.25  # must match RunACT.py / AICRobotAICControllerConfig

_DEFAULT_OUTPUT_DIR = os.environ.get("AIC_DATA_DIR", str(Path.home() / "aic_data"))


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _obs_to_state(obs: Observation) -> np.ndarray:
    cs = obs.controller_state
    tp = cs.tcp_pose
    tv = cs.tcp_velocity
    return np.array(
        [
            tp.position.x,
            tp.position.y,
            tp.position.z,
            tp.orientation.x,
            tp.orientation.y,
            tp.orientation.z,
            tp.orientation.w,
            tv.linear.x,
            tv.linear.y,
            tv.linear.z,
            tv.angular.x,
            tv.angular.y,
            tv.angular.z,
            *cs.tcp_error,
            *obs.joint_states.position[:7],
        ],
        dtype=np.float32,
    )


def _motion_to_action(motion: MotionUpdate) -> np.ndarray:
    p = motion.pose.position
    q = motion.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)


def _ros_image_to_numpy(ros_img, scale: float) -> np.ndarray:
    img = np.frombuffer(ros_img.data, dtype=np.uint8).reshape(
        ros_img.height, ros_img.width, 3
    )
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class CheatCodeDataCollector(CheatCode):
    """Runs the ground-truth CheatCode policy while recording every step.

    Requires ``ground_truth:=true`` in the evaluation launch so that TF
    frames for plugs and ports are published.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)

        try:
            self._output_dir = Path(
                parent_node.declare_parameter("output_dir", _DEFAULT_OUTPUT_DIR)
                .get_parameter_value()
                .string_value
            )
        except Exception:
            self._output_dir = Path(_DEFAULT_OUTPUT_DIR)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f"CheatCodeDataCollector: saving data to {self._output_dir}"
        )

    # ------------------------------------------------------------------
    # Policy entry point
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Ground-truth scoring helpers
    # ------------------------------------------------------------------

    def _lookup_plug_port_distance(self, task: Task) -> dict:
        """Look up ground-truth plug and port poses via /scoring/tf and
        return per-step scoring signals.  Returns empty dict on failure."""
        try:
            port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
            plug_frame = f"{task.cable_name}/{task.plug_name}_link"
            port_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                port_frame,
                Time(),
            )
            plug_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                plug_frame,
                Time(),
            )
            pp = port_tf.transform.translation
            pl = plug_tf.transform.translation
            port_pos = np.array([pp.x, pp.y, pp.z], dtype=np.float32)
            plug_pos = np.array([pl.x, pl.y, pl.z], dtype=np.float32)
            dist = float(np.linalg.norm(plug_pos - port_pos))
            pq = port_tf.transform.rotation
            port_quat = np.array([pq.x, pq.y, pq.z, pq.w], dtype=np.float32)
            return {
                "plug_pos": plug_pos,
                "port_pos": port_pos,
                "port_quat": port_quat,
                "plug_port_dist": dist,
            }
        except Exception:
            return {}

    @staticmethod
    def _read_latest_trial_score() -> dict:
        """Try to read the most recent trial score from scoring.yaml.
        Returns empty dict if unavailable."""
        results_dir = os.environ.get(
            "AIC_RESULTS_DIR", str(Path.home() / "aic_results")
        )
        score_path = Path(results_dir) / "scoring.yaml"
        if not score_path.exists():
            return {}
        try:
            with open(score_path) as f:
                data = yaml.safe_load(f)
            # Find the highest-numbered trial (most recent).
            trial_keys = [k for k in data if k.startswith("trial_")]
            if not trial_keys:
                return {}
            latest = max(trial_keys, key=lambda k: int(k.split("_")[1]))
            t = data[latest]
            return {
                "tier1": t.get("tier_1", {}).get("score", 0),
                "tier2": t.get("tier_2", {}).get("score", 0),
                "tier3": t.get("tier_3", {}).get("score", 0),
                "total": (
                    t.get("tier_1", {}).get("score", 0)
                    + t.get("tier_2", {}).get("score", 0)
                    + t.get("tier_3", {}).get("score", 0)
                ),
                "trial_key": latest,
            }
        except Exception:
            return {}

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        episode_id = str(uuid.uuid4())[:8]
        steps: list[dict] = []

        def recording_get_observation() -> Observation:
            return get_observation()

        def recording_move_robot(
            motion_update: MotionUpdate = None,
            joint_motion_update=None,
        ) -> None:
            if motion_update is not None:
                obs = get_observation()
                if obs is not None:
                    step_data = {
                        "timestamp": time.time(),
                        "state": _obs_to_state(obs),
                        "action": _motion_to_action(motion_update),
                        "images": {
                            key: _ros_image_to_numpy(
                                getattr(obs, ros_attr), _IMAGE_SCALE
                            )
                            for key, ros_attr in zip(_CAMERA_KEYS, _CAMERA_ROS_ATTRS)
                        },
                    }
                    gt = self._lookup_plug_port_distance(task)
                    if gt:
                        step_data["plug_port_dist"] = gt["plug_port_dist"]
                        step_data["port_pos"] = gt["port_pos"]
                        step_data["port_quat"] = gt["port_quat"]
                    steps.append(step_data)
            move_robot(
                motion_update=motion_update,
                joint_motion_update=joint_motion_update,
            )

        success = super().insert_cable(
            task,
            recording_get_observation,
            recording_move_robot,
            send_feedback,
        )

        # Try to capture the eval score for this trial.
        trial_score = self._read_latest_trial_score()
        if trial_score:
            self.get_logger().info(
                f"Captured eval score: {trial_score.get('total', '?')} "
                f"(tier3={trial_score.get('tier3', '?')})"
            )

        if steps:
            self._save_episode(
                episode_id, task, steps, success, trial_score=trial_score
            )
        else:
            self.get_logger().warn("No steps recorded — episode not saved.")

        return success

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_episode(
        self,
        episode_id: str,
        task: Task,
        steps: list[dict],
        success: bool,
        trial_score: dict | None = None,
    ) -> None:
        n = len(steps)
        self.get_logger().info(
            f"Saving episode {episode_id} ({n} steps, success={success})"
        )

        ep_dir = self._output_dir / "episodes" / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        # images
        for cam_key in _CAMERA_KEYS:
            cam_dir = ep_dir / "images" / cam_key
            cam_dir.mkdir(parents=True, exist_ok=True)
            for i, step in enumerate(steps):
                cv2.imwrite(str(cam_dir / f"{i:06d}.jpg"), step["images"][cam_key])

        # tabular data
        rows = []
        for i, step in enumerate(steps):
            row = {
                "episode_id": episode_id,
                "frame_index": i,
                "timestamp": step["timestamp"],
                "success": success,
                **{f"state_{j}": float(v) for j, v in enumerate(step["state"])},
                **{f"action_{j}": float(v) for j, v in enumerate(step["action"])},
                **{
                    f"image_path_{k}": str(
                        Path("episodes") / episode_id / "images" / k / f"{i:06d}.jpg"
                    )
                    for k in _CAMERA_KEYS
                },
            }
            if "plug_port_dist" in step:
                row["plug_port_dist"] = step["plug_port_dist"]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_parquet(ep_dir / "data.parquet", index=False)

        # Ground-truth port pose from first step (constant across episode)
        port_pos_list = None
        port_quat_list = None
        for step in steps:
            if "port_pos" in step:
                port_pos_list = step["port_pos"].tolist()
                port_quat_list = step["port_quat"].tolist()
                break

        # episode metadata
        meta = {
            "episode_id": episode_id,
            "num_steps": n,
            "success": success,
            "task": {
                "id": task.id,
                "cable_type": task.cable_type,
                "plug_type": task.plug_type,
                "port_type": task.port_type,
                "target_module_name": task.target_module_name,
            },
            "state_dim": 26,
            "action_dim": 7,
            "action_space": "7D pose target [x,y,z,qx,qy,qz,qw] in base_link",
            "cameras": _CAMERA_KEYS,
            "image_scale": _IMAGE_SCALE,
        }
        if port_pos_list is not None:
            meta["port_pos"] = port_pos_list
            meta["port_quat"] = port_quat_list
        if trial_score:
            meta["eval_score"] = trial_score
        with open(ep_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # global index
        index_path = self._output_dir / "dataset_index.jsonl"
        with open(index_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "num_steps": n,
                        "success": success,
                        "timestamp": time.time(),
                    }
                )
                + "\n"
            )

        self.get_logger().info(f"Episode saved → {ep_dir}")
