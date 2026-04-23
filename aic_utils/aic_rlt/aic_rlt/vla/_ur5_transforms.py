"""UR5 input/output transforms for the pi0.5 policy.

Derived from the openpi UR5 example README (openpi/examples/ur5/README.md).
Kept here rather than upstreamed into openpi to avoid modifying that repo.

UR5 action layout: 7D = [6 joint angles, 1 gripper]. Per-sample absolute;
deltas applied via DeltaActions/AbsoluteActions during training/inference.
"""

import dataclasses
import pathlib
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from typing_extensions import override

# openpi imports are lazy at call time — this module is imported from
# pi05_backend._ensure_loaded() after _setup_openpi_path() has run.


def _parse_image(img) -> np.ndarray:
    """Return uint8 (H,W,3). LeRobot sometimes stores float32 (C,H,W)."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.dtype in (np.float32, np.float64):
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


@dataclasses.dataclass(frozen=True)
class UR5Inputs:
    """Map UR5 (6 joints + 1 gripper, base+wrist cams) → pi0.5 observation."""

    EXPECTED_KEYS: ClassVar[tuple[str, ...]] = (
        "joints",
        "gripper",
        "base_rgb",
        "wrist_rgb",
    )

    def __call__(self, data: dict) -> dict:
        state = np.concatenate(
            [np.asarray(data["joints"]), np.asarray(data["gripper"])]
        )
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # No right wrist on UR5 — zero-image + mask False
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs:
    """Slice pi0.5's 32D padded output down to UR5's 7D."""

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, : self.action_dim]}


def make_ur5_data_config_cls():
    """Factory that returns a DataConfigFactory subclass wired for UR5.

    Defined as a factory (not a module-level class) so importing this module
    does not require openpi to be on sys.path. Call after _setup_openpi_path().
    """
    from openpi.training import config as openpi_config
    from openpi import transforms as _transforms

    @dataclasses.dataclass(frozen=True)
    class Pi05UR5DataConfig(openpi_config.DataConfigFactory):
        """DataConfig for pi0.5 on UR5-shape data (6 joints + 1 gripper)."""

        # No repack needed when inputs are already in {joints, gripper, base_rgb, wrist_rgb, prompt} form.
        repack_transforms: _transforms.Group = dataclasses.field(
            default_factory=_transforms.Group,
        )
        action_sequence_keys: Sequence[str] = ("actions",)
        use_delta_joint_actions: bool = True

        @override
        def create(
            self,
            assets_dirs: pathlib.Path,
            model_config,
        ):
            data_transforms = _transforms.Group(
                inputs=[UR5Inputs()],
                outputs=[UR5Outputs()],
            )
            if self.use_delta_joint_actions:
                delta_action_mask = _transforms.make_bool_mask(
                    6, -1
                )  # first 6 delta, gripper absolute
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )
            model_transforms = openpi_config.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=self.repack_transforms,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
                use_quantile_norm=True,  # pi0.5 default
            )

    return Pi05UR5DataConfig
