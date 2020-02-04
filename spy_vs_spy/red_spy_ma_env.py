import gym
import numpy as np

from multi_agent_gym import AgentEnv


class RedSpyEnv(AgentEnv):
    def __init__(self, agent_id: str, server: str):
        super().__init__(agent_id, server)

    def _init_observation_space(self) -> None:
        self.observation_space = gym.spaces.MultiDiscrete([4, 4, 4])

    def _init_action_space(self) -> None:
        self.observation_space = gym.spaces.MultiDiscrete([4, 4, 4])

    def reset(self):
        super().reset()

    def step(self, action):
        super().step(action)

    def render(self, mode='human'):
        pass

    def close(self):
        super().close()


if __name__ == '__main__':
    red_spy_env = RedSpyEnv("red1", "localhost:50051")

    red_spy_env.reset()
