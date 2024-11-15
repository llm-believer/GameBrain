from typing import Tuple
import hnswlib
import numpy as np
from game_state import GameStateManager
from visual_util import compress
from typing import Type
from abc import ABC, abstractmethod


def bit_count(bits):
  return bin(bits).count("1")


class SingleReward(ABC):
  def __init__(self, name: str, game_state_manager: GameStateManager, **kwargs):
    self.name = name
    self.game_state_manager = game_state_manager

  @abstractmethod
  def calculate(self) -> float:
    pass

  def reset(self):
    pass


class ExplorationReward(SingleReward):
  def __init__(self, name: str, game_state_manager: GameStateManager, **kwargs):
    super().__init__(name, game_state_manager, **kwargs)
    self.similar_frame_dist = kwargs["similar_frame_dist"]
    self.vec_dim = 4320
    self.num_elements = 20000  # max
    self.explore_weight = 1
    self.base_explore = 0
    self.levels_satisfied = False
    self.output_shape = (36, 40, 3)
    self.reset()

  def calculate(self):
    self.update()
    pre_rew = self.explore_weight * 0.005
    post_rew = self.explore_weight * 0.01
    cur_size = self.knn_index.get_current_count()
    base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
    post = (cur_size if self.levels_satisfied else 0) * post_rew
    return base + post

  def reset(self):
    self.init_knn()

  def init_knn(self):
    # Declaring index
    self.knn_index = hnswlib.Index(
      space="l2", dim=self.vec_dim
    )  # possible options are l2, cosine or ip
    # Initing index - the maximum number of elements should be known beforehand
    self.knn_index.init_index(max_elements=self.num_elements, ef_construction=100, M=16)

  def update(self):
    frame_vec = (
      compress(self.game_state_manager.emulator.current_frame(), self.output_shape)
      .flatten()
      .astype(np.float32)
    )
    if self.knn_index.get_current_count() == 0:
      # if index is empty add current frame
      self.knn_index.add_items(
        frame_vec, np.array([self.knn_index.get_current_count()])
      )
    else:
      # check for nearest frame and add if current
      labels, distances = self.knn_index.knn_query(frame_vec, k=1)
      if distances[0][0] > self.similar_frame_dist:
        # print(f"distances[0][0] : {distances[0][0]} similar_frame_dist : {self.similar_frame_dist}")
        self.knn_index.add_items(
          frame_vec, np.array([self.knn_index.get_current_count()])
        )


# class RewardWrapper(SingleReward):
#    def __init__(self, *, name: Optional[str] = None, fields: Optional[str]=None, game_state_manager: Optional[GameStateManager]=None, reward:Optional[SingleReward]=None, **kwargs):
#        super().__init__(name, game_state_manager, **kwargs)
#
#    def calculate(self) -> float:
#       return self.reward.calculate() * self.weight
#
#    def update(self):
#       self.reward.update()
#
#    def reset(self):
#       self.reward.reset()
#
# Useful reward wrappers
# def sum_reward(name: str, field_name: str, game_state_manager: GameStateManager):
#    class SumReward(SingleReward):
#        def calculate(self) -> float:
#            return sum(self.game_state_manager.get(field_name))
#
#    return SumReward(name, game_state_manager)


class RewardManager(object):
  def __init__(
    self, range: Tuple[float, float], config, game_state_manager: GameStateManager
  ):
    self.game_state_manager = game_state_manager
    self.range = range
    self.reward_scale = 1
    self._total_reward = 0
    self.similar_frame_dist = config["sim_frame_dist"]
    self.state_scores = {}

    self.reward_items = []
    self.add_reward(
      ExplorationReward, "explore", similar_frame_dist=self.similar_frame_dist
    )

  def add_reward(
    self, reward_type: Type[SingleReward], name: str, weight=1.0, **kwargs
  ):
    reward_item = reward_type(name, self.game_state_manager, **kwargs)
    self.reward_items.append((reward_item, weight))

  def update(self):
    for reward_item, weight in self.reward_items:
      self.state_scores[reward_item.name] = (
        self.reward_scale * reward_item.calculate() * weight
      )
    return self.total_reward

  @property
  def total_reward(self):
    self._total_reward = sum([val for _, val in self.get_reward_components().items()])
    return self._total_reward

  def get_reward_components(self):
    return self.state_scores

  def reset(self):
    for reward_item, _ in self.reward_items:
      reward_item.reset()
