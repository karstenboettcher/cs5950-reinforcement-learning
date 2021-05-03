from rlgym.utils import math, common_values
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
import math as m


class SpeedReward(RewardFunction):
    PLAYER_MAX_VEL = 2300
    PLAYER_SPEED_WEIGHT = 0.01

    def __init__(self):
        super().__init__()
        print("Using SpeedReward")

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        inst_speed = math.vecmag(player.car_data.linear_velocity / self.PLAYER_MAX_VEL)
        return inst_speed * self.PLAYER_SPEED_WEIGHT

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        final_reward = self.get_reward(player, state, previous_action)
        return final_reward
