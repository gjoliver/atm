# Main trainer.

import agents
import configs
import numpy as np
import replay_buffers
import workers


def train_loop(config, worker, replay_buffer, agent):
  step = 0
  while step < config.training_steps:
    obs, actions, rewards, next_obs, done = replay_buffer.sample_batch(
      config.batch_size)

    obs = np.vstack(obs)
    actions = np.array(actions, dtype=np.uint8)
    rewards = np.array(rewards, dtype=np.float32)
    next_obs = np.vstack(next_obs)
    done = np.hstack(done)

    step, loss = agent.step(obs, actions, rewards, next_obs, done)

    # Update target network
    if step % config.target_network_update_steps == 0:
      agent.update_target_network()

    if step % config.eval_steps == 0:
      print('Step {}, loss {}, epsilon {}'.format(
        step, loss, agent.get_epsilon()))
      print('Eval: {}'.format(worker.eval(agent, step % 1000 == 0)))

    # Queue a new episode for next batch.
    replay_buffer.add_episode(worker.episode(agent))


def main():
  config = configs.cartpole

  # Test with CartPoleV0 game.
  worker = workers.CartPole()
  replay_buffer = replay_buffers.ReplayBuffer(
    config, worker.obs_length())

  agent = agents.DQNAgent(worker.obs_length(),
                          worker.num_actions(),
                          config)

  # Seed replay buffer with some random episodes to kick off training.
  new_frames = 0
  while new_frames < config.batch_size:
    new_frames += replay_buffer.add_episode(worker.episode(agent))

  try:
    train_loop(config, worker, replay_buffer, agent)
  except KeyboardInterrupt:
    print('Training stopped, final eval: {}'.format(worker.eval(agent, True)))


if __name__ == "__main__":
  main()
