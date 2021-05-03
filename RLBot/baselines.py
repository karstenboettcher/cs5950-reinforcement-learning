import gym
import rlgym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from speed_reward import SpeedReward
from touch_ball_reward import TouchBallReward
from rlgym.utils.terminal_conditions import common_conditions
from rlgym.utils.obs_builders.default_obs import DefaultObs

conditions = [
    common_conditions.TimeoutCondition(5000),
    common_conditions.BallTouchedCondition()
]

env = rlgym.make("Duel", spawn_opponents=False, reward_fn=TouchBallReward(),
                 terminal_conditions=conditions, obs_builder=DefaultObs())

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='.\\PPO', name_prefix='ppo_bot')
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_rl_tensorboard/")
model = PPO.load('D:\\Homework\\CS 5950\\RLB\\PPO\\ppo_bot_500000_steps',
                 env=env, verbose=1, tensorboard_log="./ppo_rl_tensorboard/")
model.learn(total_timesteps=5000000000, callback=checkpoint_callback)
model.save('D:\\Homework\\CS 5950\\RLB\\PPO\\ppo_bot_final')


# model = PPO.load('D:\\Homework\\CS 5950\\RLB\\PPO\\ppo_bot_500000_steps', env=env, verbose=1)
# obs = env.reset()
# for i in range(1000000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     if done:
#         obs = env.reset()
