from abc import ABC, abstractmethod
import numpy as np
import enum


class Emulator(ABC):
  @abstractmethod
  def action_len(self) -> int:
    pass

  @abstractmethod
  def get_action(self, action: int) -> enum.Enum:
    pass

  @abstractmethod
  def reset(self):
    pass

  @abstractmethod
  def current_frame(self) -> np.ndarray:
    """
    return the current frame as a numpy array
    """
    pass

  @abstractmethod
  def run_action(self, action: int):
    pass

  @abstractmethod
  def get_last_n_frames(self, n=3) -> np.ndarray:
    """
    Return the most recent n frames as a numpy array
    """
    pass

  @abstractmethod
  def read_one_byte(self, address) -> int:
    """
    Read one byte from memory at address.
    """
    pass

  def read_memory(self, address, size, type="hex"):
    """
    Read a value from memory as hex.
    """
    result = 0
    if type == "hex":
      for i in range(size):
        result |= self.read_one_byte(address + i) << (8 * i)
    else:
      assert type == "dec", "Invalid type"
      result = self.read_memory_decimal(address, size)
    return result

  def read_bit(self, address, bit):
    return (self.read_memory(address, 1) >> bit) & 1

  def read8(self, address):
    return self.read_memory(address, 1)

  def read16(self, address):
    return self.read_memory(address, 2)

  def read_memory_decimal(self, address, size):
    """
    Read a value from memory as decimal.
    """
    result = 0
    for i in range(size):
      value = self.read_memory(address, 1)
      to_dec = 10 * ((value >> 4) & 0x0F) + (value & 0x0F)
      result += to_dec * (100**i)
    return result
