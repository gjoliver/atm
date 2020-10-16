# Hyper parameters.

class DotDict(dict):
  """Dot notation access to dictionary attributes."""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


cartpole = DotDict({
  'batch_size': 32,
  'checkpoint_steps': 1000,
  'eps_start': 1,
  'eps_end': 0.01,
  'eps_decay_steps': 100,
  'eps_decay_rate': 0.9,
  'eval_steps': 100,
  'gamma': 0.9,  # If we do n-step update, gamma needs to be updated too.
  'hidden_layer_size': 32,
  'learning_rate_initial': 0.01,
  'learning_rate_final': 0.0001,
  'learning_rate_decay_rate': 0.1,
  'learning_rate_decay_step': 1000,
  'max_step': 300,
  'num_hidden_layers': 2,
  'optimizer_epsilon': 0.00002,
  'replay_buffer_size': 100000,
  'training_steps': 100000,
  'target_network_update_steps': 100,
})


atm = DotDict({
  'batch_size': 32,
  'checkpoint_steps': 1000,
  'earliest_start_idx': 200,
  'eps_start': 0.3,
  'eps_end': 0.01,
  'eps_decay_steps': 1000,
  'eps_decay_rate': 0.95,
  'eval_steps': 100,
  'gamma': 0.95,  # If we do n-step update, gamma needs to be updated too.
  'hidden_layer_size': 500,
  'history': 60,  # Given data from last 60 days.
  'learning_rate_initial': 0.0001,
  'learning_rate_final': 0.000001,
  'learning_rate_decay_rate': 0.1,
  'learning_rate_decay_step': 1000,
  'max_step': 30,  # Max episode length.
  'num_hidden_layers': 3,
  'optimizer_epsilon': 0.00002,
  'replay_buffer_size': 100000,
  'training_steps': 100000,
  'target_network_update_steps': 100,
})
