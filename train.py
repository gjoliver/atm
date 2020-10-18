# Main trainer.

import agents
import configs
import numpy as np
import os
import replay_buffers
import tensorflow as tf
import workers

tf.get_logger().setLevel('ERROR')


def train_loop(base_dir, config, worker, replay_buffer, agent):
  progress_file = os.path.join(base_dir, 'progress.csv')
  # Reset progress file.
  with open(progress_file, 'w') as f:
    f.write('Step,Loss,Eval\n')

  chkpt_base_dir = os.path.join(base_dir, 'checkpoints')
  os.makedirs(chkpt_base_dir, exist_ok=True)
  eval_base_dir = os.path.join(base_dir, 'eval')
  os.makedirs(eval_base_dir, exist_ok=True)

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
      png_path = os.path.join(eval_base_dir, '{}.png'.format(step))
      eval = worker.eval(agent, render=True, png_path=png_path)
      print('Step {}, loss {}, epsilon {}, eval {}'.format(
        step, loss, agent.get_epsilon(), eval))
      with open(progress_file, 'a') as f:
        f.write('{},{},{}\n'.format(step, loss, eval))

    if step % config.checkpoint_steps == 0 and step > 0:
      chkpt_path = os.path.join(chkpt_base_dir, '{:010d}'.format(step))
      print('Checkpointing model to: {}'.format(chkpt_path))
      agent.checkpoint_model(chkpt_path)

    # Queue a new episode for next batch.
    replay_buffer.add_episode(worker.episode(agent))


def main():
  config = configs.atm

  # Test with CartPoleV0 game.
  worker = workers.ATM(config)
  replay_buffer = replay_buffers.ReplayBuffer(
    config, worker.obs_length())

  agent = agents.DQNAgent(worker.obs_length(),
                          worker.num_actions(),
                          config)

  base_dir = os.path.join('__out__', worker.name())
  os.makedirs(base_dir, exist_ok=True)

  # Seed replay buffer with some random episodes to kick off training.
  new_frames = 0
  while new_frames < config.batch_size:
    new_frames += replay_buffer.add_episode(worker.episode(agent))

  try:
    train_loop(base_dir, config, worker, replay_buffer, agent)
  except KeyboardInterrupt:
    png_path = os.path.join(base_dir, 'eval', 'last.png')
    print('Training stopped, final eval: {}'.format(
      worker.eval(agent, render=True, png_path=png_path)))

    chkpt_path = os.path.join(base_dir, 'checkpoints', 'last')
    print('Checkpointing model to: {}'.format(chkpt_path))
    agent.checkpoint_model(chkpt_path)


if __name__ == "__main__":
  main()
