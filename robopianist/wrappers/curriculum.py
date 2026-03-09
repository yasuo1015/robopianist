# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Curriculum wrapper for velocity reward.

Delays activation of velocity_reward until global step >= velocity_reward_start_step.
"""

from typing import Any

import dm_env
import numpy as np
from dm_env_wrappers import EnvironmentWrapper


class VelocityRewardCurriculumWrapper(EnvironmentWrapper):
    """Wrapper that sets task._velocity_reward_curriculum_step for curriculum learning.

    Call set_global_step(step) before each env.step() so the task knows when to
    enable velocity_reward (when step >= velocity_reward_start_step).
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        task: Any,
        velocity_reward_start_step: int,
    ) -> None:
        super().__init__(environment)
        self._task = task
        self._velocity_reward_start_step = velocity_reward_start_step
        self._current_step = 0

    def set_global_step(self, step: int) -> None:
        """Update the global step (call before each env.step in training loop)."""
        self._current_step = step

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        self._task._velocity_reward_curriculum_step = self._current_step
        return self._environment.step(action)
