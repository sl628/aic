from math import ceil
from time import monotonic


class OracleStage:
    stage = None

    def __init__(self, policy):
        self.policy = policy

    @property
    def frames(self):
        return self.policy.frames

    @property
    def motion(self):
        return self.policy.motion

    def begin(self):
        self.policy.set_stage(self.stage)

    def param(self, name):
        return self.policy.param(name)

    def scaled_steps(self, name, allow_zero=False):
        steps = self.param(name)
        if allow_zero and steps == 0:
            return 0
        return max(1, ceil(steps / self.policy.speed_scale()))

    def timeout_seconds(self):
        return float(self.param("oracle_stage_timeout_seconds"))

    def timeout_start(self):
        return monotonic()

    def timed_out(self, start):
        return monotonic() - start >= self.timeout_seconds()

    def timeout_error(self):
        raise TimeoutError(
            f"{self.stage} did not succeed within "
            f"{self.timeout_seconds():.1f}s wall time"
        )

    def alignment_gain(self):
        return self.param("oracle_alignment_integral_gain")

    def integrator_limit(self):
        return self.param("oracle_alignment_integrator_limit")

    def progress(self, step, steps):
        return step / steps
