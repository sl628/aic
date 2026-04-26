from copy import deepcopy
from dataclasses import dataclass

from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Transform

from ch_milestones.config.task_config import port_link_frame


@dataclass(frozen=True)
class OracleFrames:
    port_frame: str
    plug_frame: str
    port_ref: Transform

    @classmethod
    def load(cls, node, guide, task: Task):
        port_frame = port_link_frame(task)
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        node.get_logger().info(f"Targeting port frame: {port_frame}")

        guide.wait_for("base_link", port_frame)
        guide.wait_for("base_link", plug_frame)

        port_ref = guide.transform("base_link", port_frame)
        node.get_logger().info(
            "Oracle frames: using port-link reference convention "
            "orientation=port_link, position=port_link+base_z_offset"
        )
        return cls(
            port_frame=port_frame,
            plug_frame=plug_frame,
            port_ref=port_ref,
        )

    @property
    def orientation_ref(self):
        return self.port_ref

    def offset_ref(self, z_offset: float):
        target = deepcopy(self.port_ref)
        target.translation.z += z_offset
        return target
