# 为 habitat 环境中的任务提供稀疏的奖励信号。
from typing import Any

from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import Success

# SparseReward 类继承自 Measure，表明它是 habitat 环境的一个自定义测量
# 测量类根据任务的成功与否给出相应的奖励值，如果任务成功，则奖励为预先设定的成功奖励值，否则为零
@registry.register_measure
class SparseReward(Measure):
    cls_uuid: str = "sparse_reward"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self,
        *args: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                Success.cls_uuid,
            ],
        )
        self._metric = None
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        # success
        success = task.measurements.measures[Success.cls_uuid].get_metric()
        success_reward = self._config.SUCCESS_REWARD if success else 0.0
        self._metric = (
            success_reward
        )

