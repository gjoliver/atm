# Simple replay buffer.
# TODO(jungong) : maybe use Reverb if we want to do distributed training.

from collections import deque
import random

class ReplayBuffer:
  def __init__(self, config, obs_length):
    self._frames = deque(maxlen=config.replay_buffer_size)

  def add_episode(self, episode):
    for frame in episode:
      self._frames.append(frame)
    return len(episode)

  def sample_batch(self, batch_size):
    frames = random.sample(self._frames, batch_size)
    return ([obs for obs, _, _, _,_ in frames],
            [action for _, action, _, _, _ in frames],
            [reward for _, _, reward, _, _ in frames],
            [next_obs for _, _, _, next_obs, _ in frames],
            [done for _, _, _, _, done in frames])
