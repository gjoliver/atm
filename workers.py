# Workers for different tasks.

from enum import Enum
import glob
import gym
import numpy as np
import plot
import random


class Worker(object):
  def name(self):
    assert False, 'Not implemented.'

  def obs_length(self):
    assert False, 'Not implemented.'

  def num_actions(self):
    assert False, 'Not implemented.'

  def _obs_to_tensor(self, obs):
    return np.array(obs).reshape(1, self.obs_length())


class CartPole(Worker):
  def __init__(self, config):
    self._env = gym.make('CartPole-v0')
    self._max_step = config.max_step

  def __del__(self):
    self._env.close()

  def name(self):
    return 'cartpole'

  def obs_length(self):
    return self._env.observation_space.shape[0]

  def num_actions(self):
    return self._env.action_space.n

  def play(self, action_fn):
    episode = []
    step = 0
    obs = self._env.reset()
    while step < self._max_step:
      action = action_fn(obs)
      next_obs, reward, done, _ = self._env.step(action)

      episode.append([obs, action, reward, next_obs, done])

      obs = next_obs
      step += 1

      if done: break
    return episode

  def episode(self, agent, eval=False, render=False):
    def action_fn(obs):
      if render: self._env.render()
      return agent.get_action(self._obs_to_tensor(obs), eval)
    return self.play(action_fn)

  def eval(self, agent, render=False):
    return len(self.episode(agent, eval=True, render=render))


class Trade(Enum):
  BUY = 0
  HOLD = 1
  SELL = 2


class _Position(object):
  # TODO(jungong) : think about reward more.
  # Right now, always 0 reward until the end of an episode.
  def __init__(self):
    # TODO(jungong) : handle cumulative rewards.
    self.reset()

  def reset(self):
    self._position = 0  # 1: Long, -1: Short.
    self._entry_price = None

  def has_position(self):
    return self._position != 0

  def reward(self, close_price):
    if self._position == 0:
      return 0
    pl = (close_price - self._entry_price) / self._entry_price
    # Use percentage number (note, not percentage) as reward.
    # E.g., if the position is up 6%, reward is 6.
    # Also cap reward at 10% profit or loss.
    pl = np.sign(pl) * min(0.1, abs(pl)) * 100
    # If this is a short position, we should flip the reward.
    if self._position < 0:
      pl = -pl

  def action(self, action, price):
    if action == 1:  # Hold
      return 0, False

    if ((action == 0 and self._position > 0) or
        (action == 2 and self._position < 0)):
      # No change.
      return 0, False

    if action == 0 and self._position == 0:  # Open long position.
      self._position = 1
      self._entry_price = price
      return 0, False

    if action == 2 and self._position == 0:  # Open short position.
      self._position = -1
      self._entry_price = price
      return 0, False

    if ((action == 0 and self._position == -1) or  # Close short position.
        (action == 2 and self._position == 1)):    # Close long position.
      r = self.reward(price)
      self.reset()
      return r, True


class ATM(Worker):
  def __init__(self, config):
    self._history = config.history
    self._max_step = config.max_step

  def name(self):
    return 'atm'

  def obs_length(self):
    # Open, High, Low, Close, Volume, Plus 5 SMAs.
    return 10 * self._history

  def num_actions(self):
    # Buy, Hold, Sell.
    return 3

  def get_obs(self, data, idx):
    # Select self._history rows and all columns except for date.
    sub_array = data[idx - self._history:idx, 1:]
    # The first columns are Open, High, Low, Close. So current close
    # price is column 3 (0-indexed).
    cur = sub_array[-1, 3]
    return ((sub_array - cur) / cur).flatten().tolist()

  def episode_for_file(self, file, agent, eval, render):
    data = np.load(file)

    BUFFER_START = 200
    BUFFER_END = self._max_step

    # Random starting index.
    episode = []
    action_mask = [False] * self._history
    action_type = None
    start = random.randint(BUFFER_START, len(data) - BUFFER_END)
    position = _Position()

    obs = self.get_obs(data, start)
    done = False
    for i in range(self._max_step):  # At most _max_step steps.
      price = data[start + i, 4]

      # Decide what to do.
      if done:
        # Trade closed. Always hold.
        action = 1
      else:
        action = agent.get_action(obs)

      # Act.
      reward, done = position.action(action, price)
      next_obs = self.get_obs(data, start + i)

      episode.append([obs, action, reward, next_obs, done])
      obs = next_obs

      if action_type is None and action != 1:
        action_type = action
      # Whenever a position is open, action_mask should always be True.
      action_mask.append(position.has_position())

    if render:
      # TODO(jungong) : plot actions in the chart.
      plot.plot_chart(data[start - self._history:start + len(episode), :],
                      dict(mask=action_mask, type=action_type))

    return episode

  def episode(self, agent, eval=False, render=False):
    fs = glob.glob('data/train/*.npy')
    return self.episode_for_file(random.sample(fs, 1), agent, eval, render)

  def eval(self, agent, render=False):
    test_file = 'data/test/SPY.npy'
    return self.episode_for_file(test_file, agent, True, render)
