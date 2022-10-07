import time
from typing import Optional, Tuple

import gym
import numpy as np
from copy import deepcopy
from IPython.display import clear_output
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HighwayGame(gym.Env):
    def __init__(self, n_lanes: int = 3, max_steps: int = 60):
        """
        클래스의 기본적인 세팅을 초기화 합니다.
        (왠만하면) 이 클래스에서 쓰일 모든 변수는 여기서 초기화 돼야 합니다.
        Env class는 observation_space와 action_space가 필수적으로 직접 초기화 되어야 합니다.

        Parameters
        --------------------
            :param n_lanes: number of lanes for Highway game
            :param max_steps: maximum steps of environment = length of the highway
        """
        self.n_lanes = n_lanes
        self.max_steps = max_steps
        # self.road = np.load(os.path.join(PROJECT_PATH, "w1/data/road.npy"))  # 미리 준비해 둔 highway 맵
        self.road = self.generate_road()
        self.steps = 0      # 현재 steps = in-game time
        self.current_lane = n_lanes//2      # 맨 처음에 player는 lane의 중간에 위치합니다.

        self.observation_space = gym.spaces.Discrete(n_lanes)   # (0 ~ n_lane - 1) 까지의 정수
        self.action_space = gym.spaces.Discrete(n_lanes)        # (0 ~ n_lane - 1) 까지의 정수

        self.reset()    # Environment를 작동시키기 위해 초기화

    def get_state(self) -> GymObs:
        """
        현재 state를 리턴합니다.
        Returns
        -------------------
            :return: np.ndarray, 현재 state
        """
        return self.road[self.steps]

    def get_obs(self) -> GymObs:
        """
        현재 관찰 가능한 상태를 리턴합니다.
        ** state와 obs는 다릅니다! obs는 agent에게 직접 제공되는 state 중 하나로,
        """
        return self.road[min(self.steps + 1, self.max_steps - 1)]

    def get_encoded_obs(self) -> GymObs:
        """
        현재 관찰 가능한 상태를 정수 형태로 인코딩 해서 리턴합니다. (only for discrete obs space)
        :return:
        """
        return np.argmin(self.get_obs())

    def reset(self) -> GymObs:
        self.road = self.generate_road()
        self.steps = 0
        self.current_lane = self.n_lanes//2
        # return self.get_obs()
        return self.get_encoded_obs()

    def step(self, action: int, debug=False) -> GymStepReturn:
        reward, done = 0, False
        info = dict()

        self.steps += 1

        if action < 0 or action >= self.n_lanes:
            reward = -1
            done = True
            info['msg'] = "Invalid action"

        elif self.road[self.steps, action]:
            reward = -1
            done = True
            info['msg'] = "You crashed!"

        elif self.steps == self.max_steps - 1:
            reward = 1
            done = True
            info['msg'] = "Finished!"

        else:
            self.current_lane = action

        if debug:
            self.render()
            print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")

        # return self.get_obs(), reward, done, info
        return self.get_encoded_obs(), reward, done, info

    def render(self, mode="human"):
        clear_output(wait=True)
        for i, lane in enumerate(self.road[self.steps:self.steps+5]):
            repr_str = ["X" if l else " " for l in lane]
            if i == 0:
                repr_str[self.current_lane] = "O"
            print("|"+" ".join(repr_str)+"|")

    def generate_road(self) -> np.ndarray:
        """
        Highway game의 맵을 생성합니다.
        :param n_lanes: number of lanes
        :param max_steps: length of the highway
        :return: np.ndarray, shape = (max_steps, n_lanes)
        """
        road = np.ones((self.max_steps, self.n_lanes), dtype=bool)
        for i in range(self.max_steps):
            if i > 0 and road[i-1].any():
                road[i, :] = 0
            elif np.random.rand() < 0.7:
                road[i, np.random.choice(self.n_lanes)] = 0

        return road

    def play(self, policy=None):
        obs = self.reset()
        done = False
        self.render()

        while not done:
            action = policy.action(obs) if policy else self.action_space.sample()
            obs, reward, done, info = self.step(action)
            self.render()
            if done:
                print(info['msg'])
    

class HanoiGame(gym.Env):
    """
    HanoiGame은 하노이탑 퍼즐을 구현한 환경입니다.
    n개의 탑이 있을 때 한쪽 끝에 있는 n개의 원판을 다른 한쪽 끝으로 옮기는 게임입니다.
    이 때 한 번에 옮길 수 있는 원판의 개수는 1개이며, 옮기려는 원판이 옮겨질 위치에 있는 맨 위의 원판보다 작아야 합니다.
    """
    def __init__(self, n=3, n_towers=3):
        self.n = n
        self.n_towers = n_towers
        self.towers = [list(range(n, 0, -1))] + [[] for _ in range(n_towers-1)]

        self.now = 0
        self.max_steps = 2**n - 1   # this is only true for 3 towers, but we use this for maximum steps for all cases

        self.observation_space = gym.spaces.Discrete(n_towers**n)
        self.action_space = gym.spaces.Discrete(n_towers**2)
        self.reset()

    def reset(self):
        self.towers = [list(range(self.n, 0, -1))] + [[] for _ in range(self.n_towers-1)]
        self.now = 0

        return self.get_obs()

    def render(self):
        clear_output(wait=True)
        print('---'*self.n_towers)
        print(self.obs_to_str())

    def step(self, action: int, effective=True, debug=False) -> GymStepReturn:
        reward, done = 0, False
        info = dict()

        frm, to = divmod(action, self.n_towers)

        next_obs = self.get_obs()

        if not self.check_action(action):
            done = True
            reward = -100
            info['msg'] = "Invalid action"

        elif effective:
            self.towers[to].append(self.towers[frm].pop())
            self.now += 1
            next_obs = self.get_obs()

        else:
            next_obs[to].append(next_obs[frm].pop())

        if self.towers[-1] == list(range(self.n, 0, -1)):
            reward = 500
            done = True
            info['msg'] = "Finished!"

        elif self.now == self.max_steps:
            reward = -1
            done = True
            info['msg'] = "Max steps!"

        if debug:
            self.render()
            print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")

        return next_obs, reward, done, info

    def get_obs(self):
        return deepcopy(self.towers)

    def obs_to_str(self):
        render_str = ["\t|".join([str(t[i]) if i < len(t) else " " for t in self.towers]) for i in reversed(range(self.n))]
        return "\n".join(render_str)

    def check_action(self, action):
        frm, to = divmod(action, self.n_towers)
        invalid_action_conditions = [
            lambda frm_, to_: frm_ == to_,
            lambda frm_, to_: len(self.towers[frm_]) == 0,
            lambda frm_, to_: len(self.towers[to_]) > 0 and self.towers[frm_][-1] > self.towers[to_][-1]
        ]

        for condition in invalid_action_conditions:
            if condition(frm, to):
                return False
        return True

    def available_actions(self):
        for action in range(self.action_space.n):
            if self.check_action(action):
                yield action

    def available_states(self):
        for action in self.available_actions():
            yield self.step(action, effective=False)[0], action

    def play(self, policy=None, debug=False):
        obs = self.reset()
        done = False
        self.render()

        while not done:
            action = policy.action() if policy else self.action_space.sample()
            obs, reward, done, info = self.step(action)
            self.render()
            if done:
                print(info['msg'])

