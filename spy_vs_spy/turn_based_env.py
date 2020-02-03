import typing
import logging
import time
import threading

import numpy as np
import gym

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


# create file handler which logs even debug messages
# fh = logging.FileHandler('spam.log')
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logger.addHandler(fh)

class TurnBasedSubEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, sub_env_id: str, heuristic_actions: bool = False):
        super(TurnBasedSubEnv, self).__init__()

        self.sub_env_id = sub_env_id

        # If heuristic_actions is true, then the human may control the agents.
        # False means the agents are autonomous.
        self.heuristic_actions = heuristic_actions

        self._init_observation_space()
        self._init_action_space()

        self.env_waiting_for_action = None
        self.env_processing_observation = None

        self._action = None
        self._observation = None
        self._reward = None
        self._done = None
        self._info = None

        self.num_episodes = 0

    def _init_observation_space(self) -> None:
        raise NotImplementedError

    def _init_action_space(self) -> None:
        raise NotImplementedError

    def set_obs(self, observation):
        assert self.env_processing_observation

        self._observation = observation
        self.env_processing_observation = False

    def _reset(self) -> np.ndarray:
        raise NotImplementedError

    def _step(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        self.num_episodes += 1

        logger.info("Resetting environment {}. Starting episode {}.".format(self.id, self.num_episodes))

        self.env_waiting_for_action = False
        self.env_processing_observation = False

        observation = self._reset()

        logger.info("Initial observation of environment {}: {}.".format(self.id, observation))
        return observation  # reward, done, info can't be included

    def step(self, action: np.ndarray) -> [np.ndarray, float, bool, dict]:
        logger.info("Using action {} on environment {}.".format(action, self.id))

        logger.debug("Waiting to apply action.")

        while not self.env_waiting_for_action:
            time.sleep(0.1)

        observation, reward, done, info = self._step(action)

        return observation, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class TurnBasedEnv:
    def __init__(self, sub_envs: typing.List[TurnBasedSubEnv]):
        self.sub_envs = sub_envs

    def reset(self):
        self.set_observations()

    def step(self):
        step_threads = []

    def set_obs(self, observation, sub_env_id):
        sub_env = None

        for s in self.sub_envs:
            if s.sub_env_id == sub_env_id:
                sub_env = s

        assert sub_env is not None
        sub_env.set_obs(observation)

    def set_observations(self) -> None:
        raise NotImplementedError


    def turn(self):
        raise NotImplementedError
