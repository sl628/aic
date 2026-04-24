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

"""ACT Trainer for AIC Cable Insertion Task.

This package provides training infrastructure for Action Chunking Transformer (ACT)
policies on synthetic cable insertion data, as an alternative to the RLT approach.
"""

from aic_utils.aic_act_trainer.train_act import (
    create_act_config,
    create_dataset_config,
    create_training_config,
    main,
)
from aic_utils.aic_act_trainer.train_with_config import load_config_from_yaml
from aic_utils.aic_act_trainer.train_with_config import main as train_with_config_main

__all__ = [
    "create_act_config",
    "create_dataset_config",
    "create_training_config",
    "load_config_from_yaml",
    "main",
    "train_with_config_main",
]
