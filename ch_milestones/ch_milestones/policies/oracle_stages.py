from ch_milestones.policies.oracle_stage_base import OracleStage  # noqa: F401
from ch_milestones.policies.stage_approach import ApproachStage  # noqa: F401
from ch_milestones.policies.stage_coarse_align import CoarseAlignStage  # noqa: F401
from ch_milestones.policies.stage_fine_align import FineAlignStage  # noqa: F401
from ch_milestones.policies.stage_insert import InsertStage  # noqa: F401


class OracleStageSet:
    def __init__(self, policy):
        self.approach = ApproachStage(policy)
        self.coarse_align = CoarseAlignStage(policy)
        self.fine_align = FineAlignStage(policy)
        self.insert = InsertStage(policy)
