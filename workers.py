# Workers for different tasks.

import gym
import numpy as np

class CartPole(object):
  def __init__(self):
    self._env = gym.make('CartPole-v0')
    self._max_step = 300

    self._obs_length = self.obs_length()

  def __del__(self):
    self._env.close()

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

  def _obs_to_tensor(self, obs):
    return np.array(obs).reshape(1, self._obs_length)

  def episode(self, agent, eval=False, render=False):
    def action_fn(obs):
      if render: self._env.render()
      return agent.get_action(self._obs_to_tensor(obs), eval)
    return self.play(action_fn)

  def eval(self, agent, render=False):
    return len(self.episode(agent, eval=True, render=render))
