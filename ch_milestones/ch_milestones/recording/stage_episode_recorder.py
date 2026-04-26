import json
import shutil
import threading
import time
from pathlib import Path

from aic_control_interfaces.msg import MotionUpdate
from aic_model_interfaces.msg import Observation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import rclpy
from rclpy.clock import Clock
from rclpy.clock_type import ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from ch_milestones.config.policy_config import STAGES
from ch_milestones.config.task_config import task_prompt
from ch_milestones.recording.episode_trim import InsertionEventTrim
from ch_milestones.recording.lerobot_format import features, frame_from_ros


class StageEpisodeRecorder:
    def __init__(
        self,
        node,
        dataset_root: str,
        repo_id: str,
        fps: int,
        task_prompt_override: str,
        image_shape,
        use_videos: bool,
        vcodec: str,
        insertion_event_topic: str,
        trim_on_insertion_event: bool,
        post_insertion_frames: int,
        require_insertion_event: bool,
        recording_clock: str,
    ):
        self.node = node
        self.base_root = Path(dataset_root).expanduser()
        self.repo_id = repo_id
        self.fps = fps
        self.task_prompt_override = task_prompt_override
        self.image_shape = image_shape
        self.use_videos = use_videos
        self.vcodec = vcodec
        self.recording_clock = recording_clock
        self.timer_clock = {
            "wall": Clock(clock_type=ClockType.STEADY_TIME),
            "sim": node.get_clock(),
        }[recording_clock]
        self.insertion_event_topic = insertion_event_topic
        self.trim = (
            InsertionEventTrim(node, post_insertion_frames, require_insertion_event)
            if trim_on_insertion_event
            else None
        )
        self.datasets = {}
        self.current_stage = None
        self.base_task_prompt = ""
        self.observation = None
        self.action = None
        self.subscriptions = []
        self.timer = None
        self.completed = False
        self.lock = threading.Lock()

    def start(self, task):
        self.base_task_prompt = self.task_prompt_override or task_prompt(task)
        self.node.get_logger().info(f"Recording task prompt: {self.base_task_prompt}")
        datasets = {}
        try:
            for stage in STAGES:
                datasets[stage] = self.open_dataset(stage)
        except Exception:
            for dataset in datasets.values():
                dataset.finalize()
            raise
        self.datasets = datasets
        self.fps = next(iter(self.datasets.values())).meta.fps
        self.current_stage = None

        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        self.subscriptions = [
            self.node.create_subscription(
                Observation, "/observations", self.on_observation, qos
            ),
            self.node.create_subscription(
                MotionUpdate, "/aic_controller/pose_commands", self.on_action, qos
            ),
        ]
        if self.trim is not None:
            self.subscriptions.append(
                self.node.create_subscription(
                    String, self.insertion_event_topic, self.on_insertion_event, qos
                )
            )
        self.timer = self.node.create_timer(
            1.0 / self.fps, self.record_frame, clock=self.timer_clock
        )

    def open_dataset(self, stage):
        root = self.base_root / stage
        repo_id = f"{self.repo_id}_{stage}"
        if self.can_resume(root):
            return LeRobotDataset.resume(
                repo_id=repo_id, root=root, batch_encoding_size=1, vcodec=self.vcodec,
            )
        return self.create_dataset(repo_id, root)

    def can_resume(self, root):
        if not root.exists():
            return False
        if not root.is_dir():
            raise RuntimeError(f"Dataset root is not a directory: {root}")

        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            if any(root.iterdir()):
                raise RuntimeError(f"Dataset root exists without metadata: {root}")
            root.rmdir()
            return False

        info = json.loads(info_path.read_text())
        if "total_episodes" not in info:
            raise RuntimeError(f"Dataset metadata missing total_episodes: {info_path}")
        if info["total_episodes"] < 0:
            raise RuntimeError(f"Dataset metadata has invalid total_episodes: {root}")
        if info["total_episodes"] == 0:
            self.node.get_logger().warn(f"Recreating empty dataset stub at {root}")
            shutil.rmtree(root)
            return False

        episodes_dir = root / "meta" / "episodes"
        missing = [
            path
            for path in (root / "meta" / "tasks.parquet", episodes_dir)
            if not path.exists()
        ]
        if missing:
            names = ", ".join(str(path.relative_to(root)) for path in missing)
            raise RuntimeError(f"Dataset at {root} is missing metadata: {names}")
        if not any(episodes_dir.glob("*/*.parquet")):
            raise RuntimeError(
                f"Dataset at {root} has no episode metadata parquet files"
            )
        return True

    def create_dataset(self, repo_id, root):
        return LeRobotDataset.create(
            repo_id=repo_id,
            root=root,
            fps=self.fps,
            robot_type="aic_controller",
            features=features(self.image_shape, self.use_videos),
            use_videos=self.use_videos,
            batch_encoding_size=1,
            vcodec=self.vcodec,
        )

    def set_stage(self, stage):
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        with self.lock:
            self.current_stage = stage
        self.node.get_logger().info(f"Recording stage: {stage}")

    def wait_for_observation(self, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and self.observation is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.observation is None:
            raise TimeoutError("Timed out waiting for /observations")

    def on_observation(self, msg):
        with self.lock:
            self.observation = msg

    def on_action(self, msg):
        with self.lock:
            self.action = msg

    def record_frame(self):
        with self.lock:
            if self.current_stage is None or self.observation is None or self.action is None:
                return
            dataset = self.datasets[self.current_stage]
            frame = frame_from_ros(self.observation, self.action, self.image_shape)
            frame["task"] = self.stage_prompt(self.current_stage)
            dataset.add_frame(frame)

    def on_insertion_event(self, msg):
        with self.lock:
            if self.current_stage != "insert":
                self.node.get_logger().warn(
                    f"Insertion event during stage '{self.current_stage}', ignoring"
                )
                return
            self.trim.mark(self.frame_count("insert"), msg.data)

    def stage_prompt(self, stage):
        return f"{self.base_task_prompt}; stage: {stage.replace('_', ' ')}"

    def frame_count(self, stage):
        buffer = self.datasets[stage].writer.episode_buffer
        return 0 if buffer is None else buffer["size"]

    def finish(self, result):
        self.completed = result["success"]

    def stop(self):
        if self.timer is not None:
            self.node.destroy_timer(self.timer)
            self.timer = None
        for sub in self.subscriptions:
            self.node.destroy_subscription(sub)
        self.subscriptions.clear()

        with self.lock:
            if not self.datasets:
                return
            try:
                if self.completed:
                    try:
                        self.save_completed_episode()
                    except Exception:
                        for stage in STAGES:
                            self.datasets[stage].clear_episode_buffer()
                        raise
                else:
                    for stage in STAGES:
                        self.datasets[stage].clear_episode_buffer()
            finally:
                for stage in STAGES:
                    self.datasets[stage].finalize()
                self.datasets.clear()

    def save_completed_episode(self):
        for stage in STAGES:
            if not self.datasets[stage].has_pending_frames():
                raise RuntimeError(f"No frames recorded for stage '{stage}'")
        if self.trim is not None:
            self.trim.apply(self.datasets["insert"])
        for stage in STAGES:
            self.log_timing(stage)
            self.datasets[stage].save_episode()

    def log_timing(self, stage):
        count = self.frame_count(stage)
        if count == 0:
            return
        seconds = count / self.fps
        self.node.get_logger().info(
            f"Stage '{stage}': {count} frames ({seconds:.2f}s at {self.fps} fps)"
        )

    @property
    def path(self):
        return self.base_root
