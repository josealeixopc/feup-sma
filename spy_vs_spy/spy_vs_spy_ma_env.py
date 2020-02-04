import numpy as np

from multi_agent_gym import MultiAgentEnv, MultiAgentServer


class SpyVsSpyEnv(MultiAgentEnv):

    def _reset(self, agent_id) -> np.ndarray:
        if agent_id == 'red1':
            return np.array([1, 2, 3])
        else:
            raise NotImplementedError

    def _step(self, agent_id, action) -> [np.ndarray, float, bool, dict]:
        if agent_id == 'red1':
            print("Action: ".format(action))
            return np.array([3, 2, 1])
        else:
            raise NotImplementedError


if __name__ == '__main__':
    spy_vs_spy_env = SpyVsSpyEnv(['red1'])
    spy_vs_spy_server = MultiAgentServer(spy_vs_spy_env, 50051)

    spy_vs_spy_server.serve()
