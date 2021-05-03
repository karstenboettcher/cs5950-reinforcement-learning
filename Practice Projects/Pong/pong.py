import os
import gin.tf
from dopamine.discrete_domains import run_experiment

DQN_PATH = 'D:\Homework\CS 5950\Pong_Rainbow'
GAME = 'Pong'

dqn_config = """
# Hyperparameters follow Hessel et al. (2018).
import dopamine.agents.rainbow.rainbow_agent
import dopamine.discrete_domains.atari_lib
import dopamine.discrete_domains.run_experiment
import dopamine.replay_memory.prioritized_replay_buffer
import gin.tf.external_configurables

RainbowAgent.num_atoms = 51
RainbowAgent.vmax = 10.
RainbowAgent.gamma = 0.99
RainbowAgent.update_horizon = 3
RainbowAgent.min_replay_history = 20000  # agent steps
RainbowAgent.update_period = 4
RainbowAgent.target_update_period = 8000  # agent steps
RainbowAgent.epsilon_train = 0.01
RainbowAgent.epsilon_eval = 0.001
RainbowAgent.epsilon_decay_period = 250000  # agent steps
RainbowAgent.replay_scheme = 'prioritized'
RainbowAgent.tf_device = '/gpu:0'  # use '/cpu:*' for non-GPU version
RainbowAgent.optimizer = @tf.train.AdamOptimizer()

# Note these parameters are different from C51's.
tf.train.AdamOptimizer.learning_rate = 0.0000625
tf.train.AdamOptimizer.epsilon = 0.00015

atari_lib.create_atari_environment.game_name = 'Pong'
# Deterministic ALE version used in the AAAI paper.
atari_lib.create_atari_environment.sticky_actions = False
create_agent.agent_name = 'rainbow'
Runner.num_iterations = 200
Runner.training_steps = 250000  # agent steps
Runner.evaluation_steps = 125000  # agent steps
Runner.max_steps_per_episode = 27000  # agent steps

AtariPreprocessing.terminal_on_life_loss = True

WrappedPrioritizedReplayBuffer.replay_capacity = 1000000
WrappedPrioritizedReplayBuffer.batch_size = 32
""".format(GAME)

if __name__ == '__main__':
  gin.parse_config(dqn_config, skip_unknown=False)

  # train our runner
  dqn_runner = run_experiment.create_runner(DQN_PATH, schedule='continuous_train')
  print('Will train DQN agent, please be patient, may be a while...')
  dqn_runner.run_experiment()
  print('Done training!')