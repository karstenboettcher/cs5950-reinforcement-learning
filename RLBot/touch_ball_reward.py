from rlgym.utils import math, common_values
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
import math as m


class TouchBallReward(RewardFunction):
    PLAYER_TO_BALL_VEL_WEIGHT = 0.05

    def __init__(self):
        super().__init__()
        self.last_touch = None

    def reset(self, initial_state: GameState):
        self.last_touch = None

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        self.last_touch = state.last_touch
        player_ball_vel = self._get_player_ball_reward(player, state) * self.PLAYER_TO_BALL_VEL_WEIGHT
        return player_ball_vel + (player.ball_touched * 25)

    @staticmethod
    def _get_player_ball_reward(player, state):
        if player.team_num == common_values.BLUE_TEAM:
            ball = state.ball
            car = player.car_data
        else:
            ball = state.inverted_ball
            car = player.inverted_car_data

        p_vel = car.linear_velocity
        b_pos = ball.position
        p_pos = car.position

        dist = math.get_dist(b_pos, p_pos)
        vel_to_ball = math.scalar_projection(p_vel, dist)

        return vel_to_ball / 100

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        return self.get_reward(player, state, previous_action)

    def _get_dist(self, player, state):
        player_pos = player.car_data.position
        ball_pos = state.ball.position
        vec_dist = math.get_dist(ball_pos, player_pos)
        dist = m.sqrt(m.pow(vec_dist[0], 2) + m.pow(vec_dist[1], 2))  # Distance from player to ball X & Y
        return dist

    def _get_dist_reward(self, player, state):
        this_dist = self._get_dist(player, state)
        dist_reward = (self.last_dist - this_dist)
        if dist_reward < 0:
            dist_reward *= 2

        self.last_dist = this_dist

    def _get_goal_reward(self, player, state):
        os = state.orange_score
        bs = state.blue_score
        team = player.team_num

        if os != self.orange_score:
            self.orange_score = os
            if team == common_values.ORANGE_TEAM and self.last_touch == player.car_id:
                return TouchBallReward.GOAL_REWARD
            return TouchBallReward.GOAL_PUNISHMENT

        if bs != self.blue_score:
            self.blue_score = bs
            if team == common_values.BLUE_TEAM and self.last_touch == player.car_id:
                return TouchBallReward.GOAL_REWARD
            return TouchBallReward.GOAL_PUNISHMENT

        return 0
