import gym
from gym import spaces

from .spy_vs_spy_env import SpyVsSpyEnv


class RedSniperEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, spy_vs_spy_env: SpyVsSpyEnv):
        super(RedSniperEnv, self).__init__()

        self.spy_vs_spy_env = spy_vs_spy_env

        # Define action and observation space
        # They must be gym.spaces objects
        self.observation_space = spaces.MultiDiscrete([self.spy_vs_spy_env.num_grey_scouts + 1,
                                                       self.spy_vs_spy_env.num_grey_scouts + 1])
        self.action_space = spaces.Discrete(self.spy_vs_spy_env.num_grey_scouts + 1)

    def step(self, action):
        raise NotImplementedError
        # return observation, reward, done, info

    def reset(self):
        raise NotImplementedError
        # return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

