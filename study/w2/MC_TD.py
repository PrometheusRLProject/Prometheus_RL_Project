import gym
import numpy as np
from collections import defaultdict


class ValueFunction:
    def __init__(self, env):
        self.env = env
        self.values = defaultdict(lambda: defaultdict(float))

    def value(self, state, action=None):
        state = str(state)
        if action is not None:
            action = str(action)
            return self.values[state][action]
        return np.mean(list(self.values[state].values()))

    def update(self, value, state, action):
        state = str(state)
        action = str(action)
        self.values[state][action] = value


class GreedyPolicy:
    def __init__(self, env: gym.Env, value_function: ValueFunction, eps=0.1):
        self.env = env
        self.value_function = value_function
        self.eps = eps

    def action(self):
        assert isinstance(env.action_space, gym.spaces.Discrete), "Only works for discrete action spaces"
        now_state = self.env.get_obs()
        # actions = list(self.env.available_actions())
        actions = np.arange(env.action_space.n)
        values = [self.value_function.value(now_state, a) for a in actions]
        if np.random.random() > self.eps:
            max_value = np.max(values)
            actions = [a for a, v in zip(actions, values) if v == max_value]
        return actions[np.random.randint(len(actions))]

    def update(self, **kwargs):
        self.value_function = kwargs.get('value_function', self.value_function)
        self.eps = kwargs.get('eps', self.eps)

    def generate_episode(self, deterministic=False, verbose=0):
        episode = []
        obs = self.env.reset()
        done, info = False, {}
        if verbose > 1:
            self.env.render()
        while not done:
            action = self.action() if deterministic else self.env.action_space.sample()
            next_obs, reward, done, info = self.env.step(action, debug=verbose > 1)
            episode.append((obs, action, reward))
            obs = next_obs
        if verbose > 0:
            print(f'Message: {info["msg"]}')
        return episode


class MonteCarloAlgorithm:
    def __init__(self,
                 env: gym.Env,
                 value_function: ValueFunction,
                 policy: GreedyPolicy,
                 gamma: float = 1.0):
        self.env = env
        self.value_function = value_function
        self.policy = policy
        self.gamma = gamma

        self.counter = defaultdict(lambda: defaultdict(int))

    def learn(self, n_episodes, eps_schedule=None, verbose=0):
        episode_reward = 0
        for i in range(n_episodes):
            verbose_ = int(i % (n_episodes // 100) == 0) if verbose == 1 else verbose
            if verbose_:
                print(f'Episode {i}')
                print(f'Current eps: {self.policy.eps}, episode reward: {episode_reward}')
            episode = self.policy.generate_episode(deterministic=True, verbose=verbose_)
            episode_reward = sum([r for _, _, r in episode])
            self.update_value(episode)
            self.policy.update(value_function=self.value_function,
                               eps=eps_schedule(i) if eps_schedule else self.policy.eps)

    def update_value(self, episode):
        G = 0
        for obs, action, reward in reversed(episode):
            obs = str(obs)
            action = str(action)

            G = self.gamma * G + reward
            self.counter[obs][action] += 1
            prev_value = self.value_function.value(obs, action)
            self.value_function.update(prev_value + (G - prev_value) / self.counter[obs][action],
                                       state=obs,
                                       action=action)


if __name__ == '__main__':

    from study.w1.Environments import HanoiGame, HighwayGame

    env = HanoiGame()
    # env = HighwayGame()
    vf = ValueFunction(env)
    policy = GreedyPolicy(env, vf, eps=0.9)

    total_steps = 5000


    def eps_schedule(i):
        if i < total_steps * 0.6:
            return 0.9
        return 0.9 ** (1 + 0.01 * i) + 0.01


    mc = MonteCarloAlgorithm(env, vf, policy, gamma=0.9)
    mc.learn(total_steps, eps_schedule=eps_schedule, verbose=1)
    env.play(policy)
