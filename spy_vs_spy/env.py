import numpy as np

from multi_agent_gym import AgentEnv, MultiAgentEnv, agent


class RedSpyEnv(AgentEnv):
    def __init__(self, agent_id: str, server: str):
        super().__init__(agent_id, server)

    def _init_observation_space(self) -> None:
        pass

    def _init_action_space(self) -> None:
        pass

    def reset(self):
        super().reset()

    def step(self, action):
        super().step(action)

    def render(self, mode='human'):
        pass

    def close(self):
        super().close()


class SpyVsSpyEnv(MultiAgentEnv):

    def _reset(self, agent_id) -> np.ndarray:
        pass

    def _step(self, agent_id, action) -> [np.ndarray, float, bool, dict]:
        pass


if __name__ == '__main__':
    spy_vs_spy_env = SpyVsSpyEnv([])
