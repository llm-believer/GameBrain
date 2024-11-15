from reward import RewardManager, SingleReward
from game_state import GameStateManager
from typing import Tuple


def bit_count(bits):
  return bin(bits).count("1")


class EventReward(SingleReward):
  def __init__(self, name: str, game_state_manager: GameStateManager):
    super().__init__(name, game_state_manager)
    self.max_event_rew = 0

  def calculate(self):
    cur_rew = self.get_all_events_reward()
    self.max_event_rew = max(cur_rew, self.max_event_rew)
    return self.max_event_rew

  def get_all_events_reward(self):
    # adds up all event flags, exclude museum ticket
    # museum_ticket = (0xD754, 0)
    base_event_flags = 13
    return max(
      sum([bit_count(flag) for flag in self.game_state_manager.get("event_flags")])
      - base_event_flags,
      # - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
      0,
    )

  def reset(self):
    self.max_event_rew = 0


class HealthReward(SingleReward):
  def __init__(self, name: str, game_state_manager: GameStateManager):
    super().__init__(name, game_state_manager)
    self.last_health = 0
    self.died_count = 0
    self.total_healing_rew = 0
    self.party_size = 0

  def calculate(self) -> float:
    cur_health = self.read_hp_fraction()
    # if health increased and party size did not change
    if (
      cur_health > self.last_health
      and self.game_state_manager.get("party_size") == self.party_size
    ):
      if self.last_health > 0:
        heal_amount = cur_health - self.last_health
        self.total_healing_rew += heal_amount * 4
      else:
        self.died_count += 1
    self.last_health = cur_health
    self.party_size = self.game_state_manager.get("party_size")
    return self.total_healing_rew

  def read_hp_fraction(self):
    hp_sum = sum(self.game_state_manager.get("party_current_hp"))
    max_hp_sum = sum(self.game_state_manager.get("party_max_hp"))
    max_hp_sum = max(max_hp_sum, 1)
    return hp_sum / max_hp_sum

  def reset(self):
    self.last_health = 0
    self.died_count = 0
    self.total_healing_rew = 0
    self.party_size = 0


class BadgeReward(SingleReward):
  def calculate(self) -> float:
    return bit_count(self.game_state_manager.get("badges"))


class MaxOpLevelReward(SingleReward):
  def __init__(self, name: str, game_state_manager: GameStateManager):
    super().__init__(name, game_state_manager)
    self.max_opponent_level = 0

  def calculate(self) -> float:
    opponent_level = max(self.game_state_manager.get("opponent_levels")) - 5
    self.max_opponent_level = max(self.max_opponent_level, opponent_level)
    return self.max_opponent_level * 0.2


class LevelSumReward(SingleReward):
  def __init__(self, name: str, game_state_manager: GameStateManager):
    super().__init__(name, game_state_manager)
    self.max_level_rew = 0
    self.levels_satisfied = False

  def calculate(self) -> float:
    explore_thresh = 22
    scale_factor = 4
    level_sum = self.get_levels_sum()
    if level_sum < explore_thresh:
      scaled = level_sum
    else:
      scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
    self.max_level_rew = max(self.max_level_rew, scaled)
    return self.max_level_rew

  def get_levels_sum(self):
    levels = self.game_state_manager.get("party_levels")
    poke_levels = [max(level - 2, 0) for level in levels]
    return max(sum(poke_levels) - 4, 0)  # subtract starting pokemon level

  def reset(self):
    self.max_level_rew = 0
    self.levels_satisfied = False


class PokemonRedReward(RewardManager):
  def __init__(
    self, range: Tuple[float, float], config, game_state_manager: GameStateManager
  ):
    super().__init__(range, config, game_state_manager)
    self.add_reward(EventReward, "event")
    self.add_reward(HealthReward, "heal")
    self.add_reward(BadgeReward, "badge")
    self.add_reward(MaxOpLevelReward, "op_lvl")
    self.add_reward(LevelSumReward, "level")
