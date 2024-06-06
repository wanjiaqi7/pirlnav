# 定义和处理目标导航任务。它通过定义代理状态、回放动作规格和导航情节来管理任务状态，并通过任务类来执行特定任务。
# 这些定义和功能可以用来训练和评估导航任务中的智能体。
from typing import List, Optional

import os

import attr

from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask

# 定义了代理状态的规格，包括位置、旋转和传感器数据。这些状态是可选的
@attr.s(auto_attribs=True, kw_only=True)
class AgentStateSpec:
    r"""Agent data specifications that capture states of agent and sensor in replay state.
    """
    position: Optional[List[float]] = attr.ib(default=None)
    rotation: Optional[List[float]] = attr.ib(default=None)
    sensor_data: Optional[dict] = attr.ib(default=None)

# 定义了回放动作的规格，包括动作名称和代理状态
@attr.s(auto_attribs=True, kw_only=True)
class ReplayActionSpec:
    r"""Replay specifications that capture metadata associated with action.
    """
    action: str = attr.ib(default=None, validator=not_none_validator)
    agent_state: Optional[AgentStateSpec] = attr.ib(default=None)

# 继承自 NavigationEpisode，并添加了一些特定于对象导航的属性，如对象类别、参考回放、场景状态等。
@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode): 
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    reference_replay: Optional[List[ReplayActionSpec]] = None
    scene_state = None
    is_thda: Optional[bool] = False
    scene_dataset: Optional[str] = "hm3d"
    scene_dataset_config: Optional[str] = ""
    additional_obj_config_paths: Optional[List] = []
    attempts: Optional[int] = 1

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"

# 定义了特定于对象导航任务的方法和属性，如任务的重置方法。
@registry.register_task(name="ObjectNav-v2")
class ObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
    _is_episode_active: bool
    _prev_action: int
    _is_resetting: bool

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_episode_active = False
        self._is_resetting = False
    
    def reset(self, episode):
        self._is_resetting = True
        obs = super().reset(episode)
        self._is_resetting = False
        return obs
