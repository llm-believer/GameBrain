from dataclasses import dataclass
from typing import Dict, List, Union
from gb_emulator import GBEmulator
import yaml


@dataclass
class GameState:
  name: str
  description: str
  addr: Union[int, List[int]]
  size: int = 1
  type: str = "hex"


class GameStateManager(object):
  def __init__(self, emulator: GBEmulator):
    self.emulator = emulator
    self.states: Dict[str, GameState] = {}
    self.state_values: Dict[str, Union[int, List[int]]] = {}

  def add_state(
    self,
    *,
    name: str,
    description: str,
    addr: Union[int, List[int]],
    size=1,
    type: str = "hex",
  ):
    self.states[name] = GameState(name, description, addr, size, type)

  def load_config(self, config_file: str):
    with open(config_file, "r") as f:
      config = yaml.load(f, Loader=yaml.FullLoader)
      for state in config["states"]:
        if isinstance(state["addr"], str):
          beg, end = state["addr"].split("-")
          state["addr"] = list(range(int(beg, 16), int(end, 16) + 1))
        self.add_state(**state)

  def update(self):
    for state in self.states.values():
      if isinstance(state.addr, int):
        self.state_values[state.name] = self.emulator.read_memory(
          state.addr, state.size
        )
      else:
        self.state_values[state.name] = [
          self.emulator.read_memory(addr, state.size) for addr in state.addr
        ]
    self.value_valid = True

  def get(self, name) -> int:
    return self.state_values[name]
