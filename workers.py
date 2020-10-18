# Workers for different tasks.

from enum import IntEnum
import glob
import gym
import numpy as np
import plot
import random
import util


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

  def episode(self, agent, eval=False):
    def action_fn(obs):
      return agent.get_action(self._obs_to_tensor(obs), eval)
    return self.play(action_fn)

  def eval(self, agent, render=False, png_path=None):
    def action_fn(obs):
      if render: self._env.render()
      return agent.get_action(self._obs_to_tensor(obs), True)
    return len(self.play(action_fn))


class T(IntEnum):
  BUY = 0
  HOLD = 1
  SELL = 2


class PT(IntEnum):
  NO_POSITION = 0
  LONG = 1
  SHORT = 2


class _Position(object):
  # TODO(jungong) : think about reward more.
  # Right now, always 0 reward until the end of an episode.
  def __init__(self):
    self._asset = 1.0
    self._trades = []
    self._mask = []
    # TODO(jungong) : handle cumulative rewards.
    self.reset_position()

  def reset_position(self):
    self._pt = PT.NO_POSITION
    self._entry_price = None

  def asset(self):
    return self._asset

  def trades(self):
    return self._trades

  def mask(self):
    return self._mask

  def type(self):
    return self._pt

  def reward(self, close_price):
    if self._pt == PT.NO_POSITION:
      return 0
    # Scaled.
    pl = util.ScalePL(close_price, self._entry_price)
    return pl if self._pt == PT.LONG else -pl

  def pl(self, close_price):
    # Linear.
    pl = (close_price - self._entry_price) / self._entry_price
    return pl if self._pt == PT.LONG else -pl

  def action(self, action, price):
    r, done = self.action_impl(action, price)
    # Everytime there is an action, we append another position type to
    # our mask.
    self._mask.append(self.type())
    return r, done

  def action_impl(self, action, price):
    if action == T.HOLD:
      return 0, False

    if ((action == T.BUY and self._pt == PT.LONG) or
        (action == T.SELL and self._pt == PT.SHORT)):
      # No change.
      return 0, False

    # Open long position.
    if action == T.BUY and self._pt == PT.NO_POSITION:
      self._pt = PT.LONG
      self._entry_price = price
      self._trades.append(price)
      return 0, False

    # Open short position.
    if action == T.SELL and self._pt == PT.NO_POSITION:
      self._pt = PT.SHORT
      self._entry_price = price
      self._trades.append(-price)
      return 0, False

    if ((action == T.BUY and self._pt == PT.SHORT) or  # Buy to close.
        (action == T.SELL and self._pt == PT.LONG)):   # Sell to close.
      r = self.reward(price)

      self._asset *= (1.0 + self.pl(price))
      self.reset_position()
      self._trades.append(price if action == T.BUY else -price)

      return r, True

    assert False, 'Should never get here {},{}'.format(action, price)

  def force_close(self, price):
    # Close whatever position that is currently being held.
    if self._pt == PT.LONG:
      action = T.SELL
    elif self._pt == PT.SHORT:
      action = T.BUY
    else:
      action = T.HOLD
    reward, done = self.action(action, price)

    return action, reward, done


class ATM(Worker):
  def __init__(self, config):
    self._history = config.history
    self._max_step = config.max_step
    self._earliest_start_idx = config.earliest_start_idx

  def name(self):
    return 'atm'

  def obs_length(self):
    # Open, High, Low, Close, Volume, Plus 5 SMAs.
    return 10 * self._history

  def num_actions(self):
    # Buy, Hold, Sell.
    return 3

  def get_obs(self, data, idx):
    # Select self._history rows and all columns except for date,
    # which is column 0.
    # Make a copy so we don't modify the original array.
    sub_array = data[idx - self._history:idx, 1:].copy()
    # The first columns are Open, High, Low, Close. So current close
    # price is column 3 (0-indexed).
    cur = sub_array[-1, 3]

    scale_price = np.vectorize(lambda x: util.ScalePL(x, cur))
    for col in range(10):
      # Column 4 is volume, we will normalize it next.
      if col == 4: continue
      sub_array[:, col] = scale_price(sub_array[:, col])

    # Now scale volume column.
    vol_min = sub_array[:, 4].min()
    vol_max = sub_array[:, 4].max()
    scale_volume = np.vectorize(lambda x: util.ScaleLinear(x, vol_min, vol_max))
    sub_array[:, 4] = scale_volume(sub_array[:, 4])

    # Roughly scales raw price to a feature in the range of [-1.0, 1.0].
    # This is so the same network can be used on $1000 stock or $1 stock.
    return sub_array.flatten().tolist()

  def one_episode(self, action_data, position_data, action_fn):
    cur_pos = _Position()
    # Random starting index.
    episode = []
    # data only contains the period we are supposed to trade on.
    # So we know that the first trading day is on self._history row.
    obs = self.get_obs(action_data, self._history)
    for i in range(self._history, len(action_data)):
      # raw_price is used to actually conduct the trade.
      position_price = position_data[i, 4]

      # TODO(jungong) : add stop-loss.
      if i == len(action_data) - 1:
        # This is the last frame. Make sure we close whatever is open.
        action, reward, done = cur_pos.force_close(position_price)
      else:
        # Otherwise, do whatever the agent tells us to.
        action = action_fn(obs)
        reward, done = cur_pos.action(action, position_price)
      next_obs = self.get_obs(action_data, i + 1)

      episode.append([obs, action, reward, next_obs, done])
      obs = next_obs

      if done:
        if not eval:
          # Return if not in eval mode. We use a single trade for training.
          break
        else:
          # In eval mode. Reset done and continue trading.
          done = False

    # After trading finishes, we should have no position.
    assert cur_pos.type() == PT.NO_POSITION

    return episode, cur_pos

  def _load_good_data(self):
    fs = glob.glob('data/train/*.npy')
    f = random.sample(fs, 1)[0]
    data = np.load(f)
    # There may be tickers that have really short trading history.
    # So keep loading until we find a good stock.
    while len(data) < self._earliest_start_idx + self._max_step:
      data = np.load(random.sample(fs, 1)[0])
    return data

  def episode(self, agent, eval=False):
    data = self._load_good_data()

    # Pick a random point to trade.
    start = random.randint(self._earliest_start_idx,
                           len(data) - self._max_step)

    # Take out the section of the data we are going to trade on.
    data = data[start - self._history:start + self._max_step,:]

    def action_fn(obs):
      return agent.get_action(self._obs_to_tensor(obs), eval=eval)

    # Now actually generate the episode.
    # For training, trade outcome doesn't really matter, so simply
    # use log scaled price data as position data.
    episode, _ = self.one_episode(data, data, action_fn)
    return episode

  def eval(self, agent, num_days = 300, render=False, png_path=None):
    data = np.load('data/test/SPY.npy')

    # For eval, we are going to trade the hardcoded period starting
    # the 3000th row (late 2015).
    data = data[3000 - self._history:3000 + num_days,:]

    # For eval though, we want to know exactly how the position will
    # turn out in the end. So we need to load the raw prices for
    # determining position size.
    raw_data = np.load('data/test/SPY_raw.npy')
    # Same range as above.
    raw_data = raw_data[3000 - self._history:3000 + num_days,:]

    def action_fn(obs):
      return agent.get_action(self._obs_to_tensor(obs), eval=True)

    _, position = self.one_episode(data, raw_data, action_fn)

    if render:
      # To debug and see actual entries/exits, uncomment this line.
      # print(position.trades())
      plot.plot_chart(raw_data,
                      mask=[PT.NO_POSITION] * self._history + position.mask(),
                      png_path=png_path)

    return position.asset()
