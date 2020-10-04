# DQN agent.

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, layers, losses, models, optimizers

class CappedExpDecayLR(optimizers.schedules.ExponentialDecay):
  def __init__(self,
               initial_learning_rate,
               final_learning_rate,
               decay_steps,
               decay_rate):
    super().__init__(initial_learning_rate=initial_learning_rate,
                     decay_steps=decay_steps,
                     decay_rate=decay_rate,
                     staircase=False)
    self._final_learning_rate = final_learning_rate

  def __call__(self, step):
    return tf.math.maximum(self._final_learning_rate, super().__call__(step))


class DQNAgent(object):
  def __init__(self, obs_length, num_actions, config):
    self._obs_length = obs_length
    self._num_actions = num_actions
    self._config = config
    self._step = 0

    # DQN
    self._dqn = self.build_q_network()
    self._target_dqn = self.build_q_network()

  def build_q_network(self):
    """Builds a dueling DQN as a Keras model"""
    model = tf.keras.Sequential()

    # Input
    model.add(layers.Input(shape=(self._obs_length,)))
    # Hidden
    for _ in range(self._config.num_hidden_layers):
      model.add(layers.Dense(self._config.hidden_layer_size, activation='relu'))
    # Q values.
    model.add(layers.Dense(self._num_actions, activation=None))

    # Build model
    model.compile(
      optimizers.Adam(
        learning_rate=CappedExpDecayLR(self._config.learning_rate_initial,
                                       self._config.learning_rate_final,
                                       self._config.learning_rate_decay_step,
                                       self._config.learning_rate_decay_rate),
        epsilon=self._config.optimizer_epsilon),
      loss=losses.MSE)

    return model

  def get_epsilon(self):
    return max(
      self._config.eps_end,
      self._config.eps_start * pow(self._config.eps_decay_rate,
                                   self._step / self._config.eps_decay_steps))

  def get_action(self, obs, eval=False):
    # Calculate epsilon based on current step.
    eps = self.get_epsilon() if not eval else 0.0

    # With chance epsilon, take a random action
    if np.random.uniform() < eps:
      return np.random.randint(0, self._num_actions)

    # Otherwise, run the DQN.
    action = tf.math.argmax(self._dqn(obs), axis=1)

    return action.numpy()[0]

  def update_target_network(self):
    self._target_dqn.set_weights(self._dqn.get_weights())

  def step(self, obs, actions, rewards, next_obs, done):
    # Use targets to calculate loss (and use loss to calculate gradients)
    with tf.GradientTape() as tape:
      qs = self._dqn(obs)
      one_hot_actions = tf.one_hot(
        actions, self._num_actions, dtype=np.float32,
        on_value=1.0, off_value=0.0)
      q = tf.reduce_sum(qs * one_hot_actions, axis=1)

      # Calculate targets (bellman equation)
      qs_target = self._target_dqn(next_obs)
      max_next_qs = tf.reduce_max(qs_target, axis=1)

      q_target = (
        rewards + tf.where(done, 0.0, self._config.gamma) * max_next_qs)

      loss = tf.reduce_mean(self._dqn.loss(q_target, q))

      trainable_vars = self._dqn.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      self._dqn.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self._step += 1

    return self._step, loss.numpy()
