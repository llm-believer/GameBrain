from pathlib import Path
import matplotlib.pyplot as plt

from gb_emulator import GBEmulator


class ProgressTracker(object):
  def __init__(self, config, emulator: GBEmulator):
    self.emulator = emulator
    self.save_video = config["save_video"]
    self.print_rewards = config["print_rewards"]
    self.s_path = config["session_path"]
    self.save_final_state = config["save_final_state"]
    self.all_runs = []

  def save_step(
    self,
    info,
  ):
    step_count = info["step_count"]
    instance_id = info["instance_id"]
    total_reward = info["reward"]
    progress_reward = info["reward_components"]

    if self.print_rewards:
      prog_string = f"step: {step_count:6d} current action: {info['action']}"
      for key, val in progress_reward.items():
        prog_string += f" {key}: {val:5.2f}"
      prog_string += f" sum: {total_reward:5.2f}"
      print(f"\r{prog_string}", end="", flush=True)

    if step_count % 50 == 0:
      plt.imsave(
        self.s_path / Path(f"curframe_{instance_id}.jpeg"),
        self.emulator.current_frame(),
      )

  def save_finished_state(self, obs_memory, info):
    reset_count = info["reset_count"]
    total_reward = info["reward"]
    if self.print_rewards:
      print("", flush=True)
      if self.save_final_state:
        fs_path = self.s_path / Path("final_states")
        fs_path.mkdir(exist_ok=True)
        plt.imsave(
          fs_path / Path(f"frame_r{total_reward:.4f}_{reset_count}_small.jpeg"),
          obs_memory,
        )
        plt.imsave(
          fs_path / Path(f"frame_r{total_reward:.4f}_{reset_count}_full.jpeg"),
          self.emulator.current_frame(),
        )

  def save_screenshot(self, name, instance_id, total_reward, reset_count):
    ss_dir = self.s_path / Path("screenshots")
    ss_dir.mkdir(exist_ok=True)
    plt.imsave(
      ss_dir
      / Path(f"frame{instance_id}_r{total_reward:.4f}_{reset_count}_{name}.jpeg"),
      self.emulator.current_frame(),
    )
