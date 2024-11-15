from os.path import exists
from pathlib import Path
import uuid
import yaml
from game_env import create_env
from gb_emulator import GBEmulator
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

from games.pokemon_red import PokemonRedReward
import fire


def make_env(rank, env_conf, seed=0):
  """
  Utility function for multiprocessed env.
  :param env_id: (str) the environment ID
  :param num_env: (int) the number of environments you wish to have in subprocesses
  :param seed: (int) the initial seed for RNG
  :param rank: (int) index of the subprocess
  """

  def _init():
    env = create_env(env_conf, PokemonRedReward, GBEmulator)
    env.reset(seed=(seed + rank))
    return env

  set_random_seed(seed)
  return _init


def train(config):
  sess_path = Path(f"session_{str(uuid.uuid4())[:8]}")
  with open(config, "r") as f:
    env_config = yaml.load(f, Loader=yaml.FullLoader)
  env_config["session_path"] = sess_path
  ep_length = env_config["max_steps"]

  # Simple checking
  # env_checker.check_env(RedGymEnv(env_config))

  num_cpu = 20  # 64 #46  # Also sets the number of episodes per training iteration
  # env = SubprocVecEnv([rand_env for i in range(num_cpu)])
  env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
  # env = make_env(0, env_config)()

  checkpoint_callback = CheckpointCallback(
    save_freq=ep_length, save_path=sess_path, name_prefix="poke"
  )
  learn_steps = 5
  file_name = (
    "session_85b348fa/poke_983041_steps"  #'session_e41c9eff/poke_250871808_steps'
  )

  #'session_bfdca25a/poke_42532864_steps' #'session_d3033abb/poke_47579136_steps' #'session_a17cc1f5/poke_33546240_steps' #'session_e4bdca71/poke_8945664_steps' #'session_eb21989e/poke_40255488_steps' #'session_80f70ab4/poke_58982400_steps'
  print(f"\n current session: {sess_path}")
  if exists(file_name + ".zip"):
    print("\nloading checkpoint")
    model = PPO.load(file_name, env=env)
    model.n_steps = ep_length
    model.n_envs = num_cpu
    model.rollout_buffer.buffer_size = ep_length
    model.rollout_buffer.n_envs = num_cpu
    model.rollout_buffer.reset()
  else:
    model = PPO(
      "CnnPolicy",
      env,
      verbose=1,
      n_steps=ep_length,
      batch_size=512,
      n_epochs=1,
      gamma=0.999,
    )

  for i in range(learn_steps):
    model.learn(
      total_timesteps=(ep_length) * num_cpu * 1000, callback=checkpoint_callback
    )


if __name__ == "__main__":
  fire.Fire(train)
