from gb_emulator import GBEmulator
from reward import RewardManager
import visual_util
import numpy as np
import einops
import math

from gymnasium import spaces


class Observation(object):
  def __init__(self, emutator: GBEmulator, reward: RewardManager):
    self.emulator = emutator
    self.reward = reward

    self.frame_stacks = 3
    self.output_shape = (36, 40, 3)
    self.mem_padding = 2
    self.memory_height = 8
    self.col_steps = 16
    # Stack 3 frames together
    self.output_full = (
      self.output_shape[0] * self.frame_stacks
      + 2 * (self.mem_padding + self.memory_height),
      self.output_shape[1],
      self.output_shape[2],
    )

    self.recent_memory = np.zeros(
      (self.output_shape[1] * self.memory_height, 3), dtype=np.uint8
    )

  def get_obs_space(self):
    return spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

  def create_obs_mem(self):
    cur_frame = self.emulator.current_frame()
    recent_frames = self.emulator.get_last_n_frames(self.frame_stacks)
    compressed_frames = np.stack(
      [visual_util.compress(f, self.output_shape) for f in recent_frames]
    )
    cur_frame = visual_util.compress(cur_frame, self.output_shape)
    pad = np.zeros(shape=(self.mem_padding, self.output_shape[1], 3), dtype=np.uint8)
    cur_frame = np.concatenate(
      (
        self.create_exploration_memory(),  # This is for storing some states
        pad,
        self.create_recent_memory(),
        pad,
        einops.rearrange(compressed_frames, "f h w c -> (f h) w c"),
      ),
      axis=0,
    )
    return cur_frame

  def create_exploration_memory(self):
    w = self.output_shape[1]
    h = self.memory_height

    def make_reward_channel(r_val):
      col_steps = self.col_steps
      max_r_val = (w - 1) * h * col_steps
      # truncate progress bar. if hitting this
      # you should scale down the reward in group_rewards!
      r_val = min(r_val, max_r_val)
      row = math.floor(r_val / (h * col_steps))
      memory = np.zeros(shape=(h, w), dtype=np.uint8)
      memory[:, :row] = 255
      row_covered = row * h * col_steps
      col = math.floor((r_val - row_covered) / col_steps)
      memory[:col, row] = 255
      col_covered = col * col_steps
      last_pixel = math.floor(r_val - row_covered - col_covered)
      memory[col, row] = last_pixel * (255 // col_steps)
      return memory

    level, hp, explore = 1, 2, 3
    full_memory = np.stack(
      (
        make_reward_channel(level),
        make_reward_channel(hp),
        make_reward_channel(explore),
      ),
      axis=-1,
    )

    # if self.get_badges() > 0:
    #    full_memory[:, -1, :] = 255
    return full_memory

  def create_recent_memory(self):
    result = einops.rearrange(
      self.recent_memory, "(w h) c -> h w c", h=self.memory_height
    )
    return result
