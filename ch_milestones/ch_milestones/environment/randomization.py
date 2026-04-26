from dataclasses import dataclass
import random

from ch_milestones.config.scene_config import BOARD_PARTS, board_part_mappings


POSE_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
PART_FIELDS = ("translation", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class SceneSample:
    board_pose: tuple[float, float, float, float, float, float]
    board_parts: dict[str, bool | float]
    cable_pose: tuple[float, float, float, float, float, float]


class SceneRandomizer:
    def __init__(self, node):
        self.node = node
        seed = node.get_parameter("random_seed").value
        self.rng = random.Random(None if seed < 0 else seed)

    def sample(self) -> SceneSample:
        scene = SceneSample(
            board_pose=self.pose("task_board"),
            board_parts=board_part_mappings(self.node),
            cable_pose=self.pose("cable"),
        )
        if not self.param("randomize_scene"):
            return scene
        return SceneSample(
            self.randomized_pose("task_board", scene.board_pose),
            self.randomized_board_parts(scene.board_parts),
            self.randomized_pose("cable", scene.cable_pose),
        )

    def pose(self, prefix):
        return tuple(self.param(f"{prefix}_{field}") for field in POSE_FIELDS)

    def randomized_pose(self, prefix, pose):
        return tuple(
            self.random_value(f"{prefix}_{field}", value)
            for field, value in zip(POSE_FIELDS, pose, strict=True)
        )

    def randomized_board_parts(self, parts):
        randomized = dict(parts)
        for name in BOARD_PARTS:
            if not randomized[f"{name}_present"]:
                continue
            for field in PART_FIELDS:
                key = f"{name}_{field}"
                randomized[key] = self.random_value(key, randomized[key])
        return randomized

    def random_value(self, name, fallback):
        lower = self.param(f"{name}_min")
        upper = self.param(f"{name}_max")
        if lower > upper:
            raise ValueError(f"{name}_min cannot be greater than {name}_max")
        if lower == upper:
            return fallback

        distribution = str(self.param("randomization_distribution")).strip().lower()
        if distribution == "uniform":
            return self.rng.uniform(lower, upper)
        if distribution in ("normal", "gaussian"):
            return self.truncated_normal(lower, upper, fallback)
        raise ValueError("randomization_distribution must be 'uniform' or 'normal'")

    def truncated_normal(self, lower, upper, fallback):
        mean = min(max(float(fallback), float(lower)), float(upper))
        stddevs = float(self.param("randomization_normal_stddevs"))
        if stddevs <= 0.0:
            raise ValueError("randomization_normal_stddevs must be greater than 0")

        spread = max(abs(mean - lower), abs(upper - mean))
        if spread <= 0.0:
            return fallback

        sigma = spread / stddevs
        max_attempts = int(self.param("randomization_normal_max_attempts"))
        if max_attempts <= 0:
            raise ValueError("randomization_normal_max_attempts must be greater than 0")

        value = mean
        for _ in range(max_attempts):
            value = self.rng.gauss(mean, sigma)
            if lower <= value <= upper:
                return value
        return min(max(value, lower), upper)

    def param(self, name):
        return self.node.get_parameter(name).value
