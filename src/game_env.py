from typing import Type
import uuid
from pathlib import Path


from gymnasium import Env, spaces

from emulator import Emulator
from game_state import GameStateManager
from progress_tracker import ProgressTracker
from reward import RewardManager
from observation import Observation


class GameEnv(Env):
  def __init__(
    self,
    emulator: Emulator,
    game_state_manager: GameStateManager,
    reward_manager: RewardManager,
    config=None,
  ):
    self.max_steps = config["max_steps"]

    self.instance_id = (
      str(uuid.uuid4())[:8] if "instance_id" not in config else config["instance_id"]
    )
    Path(config["session_path"]).mkdir(exist_ok=True)
    self.current_action = emulator.get_action(0)
    self.emulator = emulator
    self.progress_tracker = ProgressTracker(config, self.emulator)
    self.game_state_manager = game_state_manager
    self.reward_manager = reward_manager
    self.obs = Observation(self.emulator, self.reward_manager)
    self.reset_count = 0
    self.current_reward = 0.0
    self.step_limit_reach = False

    # Set this in SOME subclasses
    self.metadata = {"render.modes": []}
    self.reward_range = self.reward_manager.range

    # Set these in ALL subclasses
    self.action_space = spaces.Discrete(self.emulator.action_len())
    self.observation_space = self.obs.get_obs_space()

    self.reset()

  def info(self):
    return {
      "instance_id": self.instance_id,
      "reset_count": self.reset_count,
      "step_count": self.step_count,
      "reward": self.current_reward,
      "step_limit_reach": self.step_limit_reach,
      "reward_components": self.reward_manager.get_reward_components(),
      "action": self.current_action.name.ljust(10),
    }

  def reset(self, seed=None):
    self.seed = seed
    self.emulator.reset()
    self.reward_manager.reset()

    self.step_count = 0
    self.reset_count += 1
    self.current_reward = 0.0
    self.step_limit_reach = False
    return self.obs.create_obs_mem(), self.info()

  def render(self):
    return self.emulator.current_frame()

  def step(self, action):
    self.current_action = self.emulator.get_action(action)
    self.emulator.run_action(action)
    self.game_state_manager.update()
    old_reward = self.current_reward
    self.current_reward = self.reward_manager.update()
    if self.current_reward - old_reward < 0:
      print(f"Reward drop from {old_reward} to {self.current_reward}!")

    obs_memory = self.obs.create_obs_mem()

    self.step_limit_reach = self.step_count >= self.max_steps
    self.progress_tracker.save_step(
      self.info(),
    )

    if self.step_limit_reach:
      self.progress_tracker.save_finished_state(
        obs_memory,
        self.info(),
      )

    self.step_count += 1

    return (
      obs_memory,
      self.current_reward - old_reward,
      False,
      self.step_limit_reach,
      self.info(),
    )


def create_env(config, reward_type: Type[RewardManager], emulator: Type[Emulator]):
  emulator = emulator(config)
  game_state_manager = GameStateManager(emulator)
  game_state_manager.load_config(config["game_state"])
  game_state_manager.update()
  reward_manager = reward_type((0, 15000), config, game_state_manager)
  return GameEnv(emulator, game_state_manager, reward_manager, config)
