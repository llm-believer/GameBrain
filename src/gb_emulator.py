from pyboy import PyBoy
from pyboy.utils import WindowEvent
import enum
import numpy as np
from emulator import Emulator


class GBAction(enum.Enum):
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3
  A = 4
  B = 5
  # START = 6
  # SELECT = 7


def action_to_window_event(action: GBAction) -> tuple[WindowEvent, WindowEvent]:
  if action == GBAction.UP:
    return (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP)
  elif action == GBAction.DOWN:
    return (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)
  elif action == GBAction.LEFT:
    return (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT)
  elif action == GBAction.RIGHT:
    return (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)
  elif action == GBAction.A:
    return (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A)
  elif action == GBAction.B:
    return (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B)
  # elif action == GBAction.START:
  #    return (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START)
  # elif action == GBAction.SELECT:
  #    return (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT)
  else:
    raise ValueError("Invalid action")


class GBEmulator(Emulator):
  def __init__(self, config):
    head = "headless" if config["headless"] else "SDL2"

    self.pyboy = PyBoy(
      config["gb_path"],
      debugging=False,
      disable_input=False,
      window_type=head,
      hide_window=True,
    )
    self.init_state = config["init_state"]
    self.act_freq = config["action_freq"]
    if not config["headless"]:
      print("Here")
      self.pyboy.set_emulation_speed(6)
    self.save_n_frames = 3  # Make it a config
    self._current_frame = None
    self.last_n_frames = np.zeros((self.save_n_frames, 144, 160, 3), dtype=np.uint8)

  def action_len(self) -> int:
    return len(GBAction)

  def load_state(self, state_file):
    with open(state_file, "rb") as f:
      self.pyboy.load_state(f)

  def reset(self):
    self.load_state(self.init_state)

  def _get_screen_pixels(self):
    return self.pyboy.botsupport_manager().screen().screen_ndarray()

  def current_frame(self):
    if self._current_frame is None:
      self._current_frame = self._get_screen_pixels()
    return self._current_frame

  def tick(self, n=1):
    for _ in range(n):
      self.pyboy.tick()

  def get_action(self, action: int) -> enum.Enum:
    return GBAction(action)

  def run_action(self, action: int):
    press, release = action_to_window_event(self.get_action(action))
    self.pyboy.send_input(press)
    # disable rendering when we don't need it
    # if not self.save_video and self.headless:
    #    self.pyboy._rendering(False)
    self.tick(self.act_freq)
    self.pyboy.send_input(release)
    self._current_frame = self._get_screen_pixels()
    self.last_n_frames = np.roll(self.last_n_frames, 1, axis=0)
    self.last_n_frames[0] = self._current_frame

  def get_last_n_frames(self, n=3):
    return self.last_n_frames[:n]

  def read_one_byte(self, address) -> int:
    return self.pyboy.get_memory_value(address)
