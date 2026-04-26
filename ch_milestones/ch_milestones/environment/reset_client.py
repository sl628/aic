from std_srvs.srv import Trigger

from ch_milestones.environment.services import call


class ResetClient:
    def __init__(self, node):
        self.node = node
        self.reset_client = node.create_client(Trigger, "/ch_milestones/reset_episode")
        self.clear_client = node.create_client(
            Trigger, "/ch_milestones/clear_environment"
        )

    def reset(self, timeout_sec):
        response = call(
            self.node,
            self.reset_client,
            Trigger.Request(),
            timeout_sec,
            "reset episode",
        )
        if not response.success:
            raise RuntimeError(response.message)

    def clear(self, timeout_sec):
        response = call(
            self.node,
            self.clear_client,
            Trigger.Request(),
            timeout_sec,
            "clear environment",
        )
        if not response.success:
            raise RuntimeError(response.message)
