import numpy as np
import gym
import logging
import time
import typing

from threading import Barrier
from multi_agent_gym import MultiAgentEnv, MultiAgentServer, AgentEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NUM_GRAY_SCOUTS = 5


class RedSniperEnv(AgentEnv):

    def _init_observation_space(self) -> None:
        self.observation_space = gym.spaces.MultiDiscrete([2, NUM_GRAY_SCOUTS + 1])

    def _init_action_space(self) -> None:
        self.action_space = gym.spaces.Discrete(NUM_GRAY_SCOUTS + 1)

    def render(self, mode='human'):
        pass


class RedSpyEnv(AgentEnv):

    def _init_observation_space(self) -> None:
        self.observation_space = gym.spaces.MultiDiscrete([2, NUM_GRAY_SCOUTS + 1])

    def _init_action_space(self) -> None:
        self.action_space = gym.spaces.Discrete(NUM_GRAY_SCOUTS + 1)

    def render(self, mode='human'):
        pass


class SpyVsSpyEnv(MultiAgentEnv):

    def __init__(self) -> None:
        super().__init__()
        self.num_gray_scouts = NUM_GRAY_SCOUTS
        self.blue_scout_position = None

        self.reset_barrier = Barrier(2, timeout=5)
        self.step_barrier = Barrier(2, timeout=5)

        self.red_spy_acted = False
        self.red_spy_action = None

        self.red_sniper_acted = False
        self.red_sniper_action = None

    def set_blue_scout_position(self):
        available_positions = self.num_gray_scouts + 1
        self.blue_scout_position = np.random.randint(available_positions)
        logger.info("Blue Scout position is: {}".format(self.blue_scout_position))

    def _set_agent_envs_ids(self):
        self.agent_envs_ids = ['red-sniper', 'red-spy']

    def _compute_dead_and_reward(self):
        if self.red_sniper_action == self.blue_scout_position:
            blue_scout_dead = True
            reward = 1
        else:
            blue_scout_dead = False
            reward = -1

        return blue_scout_dead, reward

    def _reset(self, agent_id) -> np.ndarray:
        if agent_id == 'red-spy':

            self.red_spy_action = None
            self.red_spy_acted = False

            self.reset_barrier.wait()

            self.set_blue_scout_position()

            return np.array([0, self.blue_scout_position])

        elif agent_id == 'red-sniper':

            self.red_sniper_action = None
            self.red_sniper_acted = False

            self.reset_barrier.wait()

            while not self.red_spy_acted:
                time.sleep(0.1)

            return np.array([0, self.red_spy_action])

        else:
            raise NotImplementedError

    def _step(self, agent_id, action) -> typing.Tuple[np.ndarray, float, bool, dict]:
        if agent_id == 'red-spy':
            logger.info("Action from RedSpy: {}".format(action))

            self.red_spy_action = action
            self.red_spy_acted = True

            while not self.red_sniper_acted:
                time.sleep(0.1)

            blue_scout_dead, reward = self._compute_dead_and_reward()
            obs = np.array([int(blue_scout_dead), self.blue_scout_position])

            self.step_barrier.wait()

            return obs, reward, True, {}

        elif agent_id == 'red-sniper':
            logger.info("Action from RedSniper: {}".format(action))

            self.red_sniper_action = action
            self.red_sniper_acted = True

            blue_scout_dead, reward = self._compute_dead_and_reward()
            obs = np.array([int(blue_scout_dead), self.red_spy_action])

            self.step_barrier.wait()

            return obs, reward, True, {}
        else:
            raise NotImplementedError


if __name__ == '__main__':
    spy_vs_spy_env = SpyVsSpyEnv()

    spy_vs_spy_server = MultiAgentServer(spy_vs_spy_env, 50051)

    spy_vs_spy_server.serve()
