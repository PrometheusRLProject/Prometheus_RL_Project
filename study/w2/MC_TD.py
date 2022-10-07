import gym
import numpy as np

from tqdm import tqdm
from typing import Callable, Optional


class ValueFunction:
    def __init__(self, observation_space: gym.spaces.Discrete, action_space: gym.spaces.Discrete):
        """
        Value function은 row: observation, column: action으로 구성된 2차원 배열입니다.

        Parameters
        --------------------------------
        :param observation_space: environment의 observation space
        :param action_space: environment의 action space
        """
        self.values = np.zeros((observation_space.n, action_space.n))

    def value(self, state: int, action: int = None) -> float:
        """
        value function으로 부터 특정 state, action의 value를 반환합니다.
        만약 action이 None이면, 해당 state의 모든 action에 대한 value의 평균을 반환합니다.
        """
        if action is not None:
            return self.values[state, action]
        return np.mean(self.values[state])

    def update(self, value: float, state: int, action: int):
        """
        value function을 특정 state, action에 대해 value로 업데이트합니다.

        Parameters
        --------------------------------
        :param value: 업데이트 할 value
        :param state: 업데이트 state
        :param action: 업데이트 action
        """
        self.values[state, action] = value


class GreedyPolicy:
    def __init__(self, env: gym.Env, vf: ValueFunction, eps: float = 0.1):
        """
        Greedy Policy는 주어진 state에서 가장 높은 value를 주는 action을 선택합니다.
        이 때, epsilon이 클 수록 greedy하지 않는 action을 무작위로 선택할 확률이 커집니다. (exploration에 대해서 나중에 다루겠습니다.)

        Parameters
        --------------------------------
        :param env: policy 대상 environment
        :param vf: policy가 사용할 value function
        :param eps: epsilon for exploration
        """
        self.env = env
        self.vf = vf
        self.eps = eps

    def action(self, state: int, deterministic=True) -> int:
        """
        주어진 state에서 greedy한 action을 반환합니다.

        Parameters
        --------------------------------
        :param state: 주어진 state
        :param deterministic: weather to use epsilon-greedy policy
        :return: greedy한 action
        """
        assert isinstance(self.env.action_space, gym.spaces.Discrete), "Action space should be discrete"
        values = self.vf.values[state]      # state에 대한 모든 action의 value

        if np.random.rand() > self.eps or deterministic:    # greedy selection
            return np.random.choice(np.argwhere(values == np.max(values)).flatten())
        return np.random.choice(self.env.action_space.n)    # random selection

    def update(self, **kwargs):
        self.eps = kwargs.get('eps', self.eps)


class PolicyEvaluation:
    def __init__(self,
                 policy: GreedyPolicy,
                 eps_schedule: Optional[Callable[[int], float]] = None):
        """
        Policy Evaluation은 주어진 policy를 사용해서 value function을 구성합니다.

        Parameters
        --------------------------------
        :param policy: 사용할 policy
        :param eps_schedule: epsilon을 조절해주는 함수 (lr scheduler와 비슷합니다.)
        """
        self.policy = policy
        self.eps_schedule = eps_schedule

    def evaluate(self, n_episodes: int = 100, steps: int = 1):
        """
        주어진 policy를 사용해서 value function을 구성합니다.

        Parameters
        --------------------------------
        :param n_episodes: 총 학습할 episode 수
        :param steps:
            실제 reward를 사용할 step 수
            MC는 episode의 마지막까지 실제 reward를 받으므로 steps=infinity,
            TD는 특정 step까지만 실제 reward를 받으므로 steps=n>0,
            TD-lambda는 one-step reward를 사용하므로 steps=1

            이 때, steps= < 0이면, steps=infinity로 간주합니다.

        :return:
        """
        rewards = []    # 각 episode의 reward 총합을 저장할 list
        progress = tqdm(range(n_episodes), total=n_episodes)    # progress bar
        for prog in progress:
            episode_buffer = []     # 해당 episode에서 step을 저장할 버퍼
            episode_reward = 0      # 해당 episode에서 reward 총합을 계산할 변수
            step_count = 0          # 몇 step 진행됐는지 기록할 변수

            # 일반적인 gym.Env의 step 루틴
            obs = self.policy.env.reset()
            done = False
            while not done:
                action = self.policy.action(obs, deterministic=False)       # policy로부터 action을 선택
                next_obs, reward, done, info = self.policy.env.step(action) # action을 취하고 one-step을 받음
                episode_buffer.append((obs, action, reward, next_obs, done))    # step을 버퍼에 저장
                obs = next_obs
                episode_reward += reward
                rewards.append(episode_reward)

                step_count += 1
                if 0 < steps <= step_count or done:     # steps까지 step을 모으거나, episode가 끝나면 value function update 시작
                    self.value_update(episode_buffer)   # value function update
                    if self.eps_schedule is not None:
                        self.policy.update(eps=self.eps_schedule(prog))     # epsilon을 조절해주는 함수가 있다면, update
                    episode_buffer.clear()

            progress.desc = f"Episode reward: {episode_reward:.2f}\t eps: {self.policy.eps:.3f}"
        return rewards

    def value_update(self, episode_buffer):
        """
        step들을 모은 episode_buffer로부터 value function을 update합니다.
        이는 하위 클래스에서 구현되어야 합니다.

        Parameters
        --------------------------------
        :param episode_buffer: step들을 저장한 버퍼
        """
        raise NotImplementedError


class TemporalDifferenceEvaluation(PolicyEvaluation):
    def __init__(self,
                 policy: GreedyPolicy,
                 eps_schedule: Optional[Callable[[int], float]] = None,
                 gamma: float = 0.9,
                 alpha: float = 0.1):
        """
        n-step TD

        Parameters
        --------------------------------
        :param gamma: discount factor
        :param alpha: value function update rate
        """
        super().__init__(policy, eps_schedule)
        self.gamma = gamma
        self.alpha = alpha

    def value_update(self, episode_buffer):
        last_episode = episode_buffer[-1]
        G = self.policy.vf.value(last_episode[3]) * (1-last_episode[-1])    # return value
        for obs, action, reward, next_obs, done in reversed(episode_buffer):
            G = reward + self.gamma * G     # return value를 계산
            prev_value = self.policy.vf.value(obs, action)
            new_value = prev_value + self.alpha * (G - prev_value)  # value function update 식
            self.policy.vf.update(new_value, obs, action)       # 계산 결과로 value function 업데이트


class TDLambda(PolicyEvaluation):
    def __init__(self,
                 policy: GreedyPolicy,
                 eps_schedule: Optional[Callable[[int], float]] = None,
                 gamma: float = 0.9,
                 lambda_: float = 0.9,
                 alpha: float = 0.1):
        super().__init__(policy, eps_schedule)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.alpha = alpha

        self.trace = np.zeros_like(self.policy.vf.values)   # eligibility trace

    def value_update(self, episode_buffer):
        last_episode = episode_buffer[-1]
        G = self.policy.vf.value(last_episode[3]) * (1-last_episode[-1])
        for obs, action, reward, next_obs, done in reversed(episode_buffer):
            G = reward + self.gamma * G
            self.trace[obs, action] += 1        # Frequency heuristic

            prev_value = self.policy.vf.value(obs, action)
            new_value = prev_value + self.alpha * (G - prev_value) * self.trace[obs, action]
            self.policy.vf.update(new_value, obs, action)

            self.trace *= self.gamma * self.lambda_    # Recency heuristic
