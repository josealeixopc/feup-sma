import logging
import time
import typing
from threading import Barrier

import gym
import numpy as np

from multi_agent_gym import MultiAgentEnv, MultiAgentServer, AgentEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NUM_GRAY_SCOUTS = 19
NUM_DISCRETE_VALUES_UNIQUENESS = 10


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


class RedSpyUniquenessObservationEnv(RedSpyEnv):

    def _init_observation_space(self) -> None:
        self.observation_space = gym.spaces.MultiDiscrete([2, NUM_GRAY_SCOUTS + 1, NUM_DISCRETE_VALUES_UNIQUENESS])


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
        logger.info("Blue Scout position is: {} of {} possible.".format(self.blue_scout_position, NUM_GRAY_SCOUTS))

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

    def _reset_red_spy(self):
        self.red_spy_action = None
        self.red_spy_acted = False

        self.reset_barrier.wait()

        self.set_blue_scout_position()

        return np.array([0, self.blue_scout_position])

    def _reset_red_sniper(self):
        self.red_sniper_action = None
        self.red_sniper_acted = False

        self.reset_barrier.wait()

        timeout = 2
        timeout_start = time.time()
        while not self.red_spy_acted:
            time.sleep(0.1)

            if time.time() > timeout_start + timeout:
                # Timeout in case it was the last episode and the RedSpy will not act anymore.
                self.red_spy_action = 0
                break

        return np.array([0, self.red_spy_action])

    def _reset(self, agent_id) -> np.ndarray:
        if agent_id == 'red-spy':
            return self._reset_red_spy()
        elif agent_id == 'red-sniper':
            return self._reset_red_sniper()
        else:
            raise NotImplementedError

    def _step_red_spy(self, action):
        logger.info("Action from RedSpy: {}".format(action))

        self.red_spy_action = action
        self.red_spy_acted = True

        while not self.red_sniper_acted:
            time.sleep(0.1)

        blue_scout_dead, reward = self._compute_dead_and_reward()
        obs = np.array([int(blue_scout_dead), self.blue_scout_position])

        self.step_barrier.wait()

        return obs, reward, True, {}

    def _step_red_sniper(self, action):
        logger.info("Action from RedSniper: {}".format(action))

        self.red_sniper_action = action
        self.red_sniper_acted = True

        blue_scout_dead, reward = self._compute_dead_and_reward()
        obs = np.array([int(blue_scout_dead), self.red_spy_action])

        self.step_barrier.wait()

        return obs, reward, True, {}

    def _step(self, agent_id, action) -> typing.Tuple[np.ndarray, float, bool, dict]:
        if agent_id == 'red-spy':
            return self._step_red_spy(action)
        elif agent_id == 'red-sniper':
            return self._step_red_sniper(action)
        else:
            raise NotImplementedError


class SpyVsSpyUniquenessObservationEnv(SpyVsSpyEnv):

    def __init__(self) -> None:
        super().__init__()
        self.red_spy_action_distribution = {}
        self._init_red_spy_action_distribution()

    def _init_red_spy_action_distribution(self):
        for i in range(NUM_GRAY_SCOUTS + 1):
            self.red_spy_action_distribution[i] = [0] * (NUM_GRAY_SCOUTS + 1)

    # def _reset_red_spy(self):
    #     obs = super()._reset_red_spy()
    #     uniqueness_average_discrete = self._calculate_uniqueness_discrete()
    #
    #     obs = np.append(obs, uniqueness_average_discrete)
    #     return obs
    #

    def _step_red_spy(self, action):
        self.red_spy_action_distribution[self.blue_scout_position][int(action)] += 1

        # uniqueness_average_discrete = self._calculate_uniqueness_discrete()
        # obs = np.append(obs, uniqueness_average_discrete)

        return super()._step_red_spy(action)

    def _calculate_uniqueness_average(self):
        max_uniqueness_values = []

        for key in self.red_spy_action_distribution:
            max_num_times_action_taken_for_obs = max(self.red_spy_action_distribution[key])
            num_times_action_taken = sum(self.red_spy_action_distribution[key])

            if num_times_action_taken == 0:
                continue

            maximum_uniqueness = max_num_times_action_taken_for_obs / num_times_action_taken

            if maximum_uniqueness != 0:
                max_uniqueness_values.append(maximum_uniqueness)

        if len(max_uniqueness_values) > 0:
            average_uniqueness = sum(max_uniqueness_values) / len(max_uniqueness_values)
        else:
            average_uniqueness = 0

        return average_uniqueness

    def _compute_dead_and_reward(self):
        blue_scout_dead, kill_reward = super()._compute_dead_and_reward()
        reward = kill_reward * 0.5 + self._calculate_uniqueness_average() * 0.5

        return  blue_scout_dead, reward

    # def _calculate_uniqueness_discrete(self):
    #     uniqueness_average = self._calculate_uniqueness_average()  # between 0 and 1
    #
    #     uniqueness_average_bin = utils.find_bin_equal_width(uniqueness_average,
    #                                                         0, 1,
    #                                                         NUM_DISCRETE_VALUES_UNIQUENESS)
    #     uniqueness_average_discrete = uniqueness_average_bin - 1  # Discrete values start at 0
    #
    #     return uniqueness_average_discrete


if __name__ == '__main__':
    spy_vs_spy_env = SpyVsSpyEnv()

    spy_vs_spy_server = MultiAgentServer(spy_vs_spy_env, 50051)

    spy_vs_spy_server.serve()
