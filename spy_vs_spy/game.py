import typing
import logging

from drl.dqn import DQNAgent
from gym import spaces

from mesa import Agent, Model
import numpy as np

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

# If HEURISTIC is not zero, then the human may control the agents. 0 means the agents are autonomous.
HEURISTIC = 0


# Rulebook for defining rules: https://cdn.1j1ju.com/medias/af/08/f4-the-werewolves-of-millers-hollow-rulebook.pdf
# Using the Basic Card Mix variant for 8 Players, with 2 Werewolves, 1 Fortune Teller and 5 Townsfolk

class Player(Agent):
    def __init__(self, unique_id, model, observation_space: spaces.MultiDiscrete, action_space: spaces.Discrete):
        super().__init__(unique_id, model)
        self.observation_space = observation_space
        self.action_space = action_space

        self.obs = None
        self.action = None

        self.obs_flattened = None

        self.num_states = observation_space.shape[0]
        self.num_actions = action_space.n

        self.drl_agent = DQNAgent(self.num_states, self.num_actions)
        self.batch_size = 16

    def step(self):
        logger.info("{} observation: {}".format(self.unique_id, self.obs))
        assert self.observation_space.contains(self.obs)

        self.action = self.get_action()
        logger.info("{} has chosen action: {}".format(self.unique_id, self.action))

    def post_step(self, new_state, reward, done):

        self.obs_flattened = np.reshape(self.obs, [1, self.num_states])
        new_state_flattened = np.reshape(new_state, [1, self.num_states])

        self.drl_agent.memorize(self.obs_flattened, self.action, reward, new_state_flattened, done)

        if len(self.drl_agent.memory) > self.batch_size:
            self.drl_agent.replay(self.batch_size)

    def get_action(self):
        self.obs_flattened = np.reshape(self.obs, [1, self.num_states])

        if HEURISTIC:
            logger.warning("Using heuristic mode.")
            return self._get_action_heuristic(self.obs_flattened)
        else:
            return self._get_action_autonomous(self.obs_flattened)

    def _get_action_autonomous(self, obs):
        action = self.drl_agent.act(obs)
        assert self.action_space.contains(action)

        return action

    def _get_action_heuristic(self, obs):
        logger.info("Enter a number from 0 to {}:".format(self.action_space.n - 1))
        action = int(input())
        assert self.action_space.contains(action)

        return action


class RedSpy(Player):
    def __init__(self, unique_id, model, observation_space, action_space):
        super().__init__(unique_id, model, observation_space, action_space)


class BlueSpy(Player):
    def __init__(self, unique_id, model, observation_space, action_space):
        super().__init__(unique_id, model, observation_space, action_space)


class RedSniper(Player):
    def __init__(self, unique_id, model, observation_space, action_space):
        super().__init__(unique_id, model, observation_space, action_space)


class BlueScout(Player):
    def __init__(self, unique_id, model, observation_space, action_space):
        super().__init__(unique_id, model, observation_space, action_space)

        self.position = None


class SpyVsSpy(Model):
    def __init__(self, num_gray_scouts: int):
        super().__init__()

        self.num_gray_scouts = num_gray_scouts

        self.red_spy = RedSpy("RedSpy",
                              self,
                              spaces.MultiDiscrete([1, self.num_gray_scouts + 1]),
                              spaces.Discrete(self.num_gray_scouts + 1))

        # self.blue_spy = BlueSpy(2, self, 1, self.num_gray_scouts + 1)

        self.red_sniper = RedSniper("RedSniper",
                                    self,
                                    spaces.MultiDiscrete([1, self.num_gray_scouts + 1]),
                                    spaces.Discrete(self.num_gray_scouts + 1))

        self.blue_scout = BlueScout("BlueScout",
                                    self,
                                    spaces.MultiDiscrete([1]),
                                    spaces.Discrete(1))

        self.players = [self.red_spy,
                        self.red_sniper,
                        # self.blue_spy,
                        self.blue_scout]

        self.num_steps = 0

    def step(self):
        logger.info("We're starting turn '{}'.".format(self.num_steps + 1))
        blue_scout_dead = False

        # Set blue scout position among all the other scouts
        self.set_blue_scout_position()

        self.red_spy.obs = np.array([int(blue_scout_dead), self.blue_scout.position])
        self.red_spy.step()

        self.red_sniper.obs = np.array([int(blue_scout_dead), self.red_spy.action])
        self.red_sniper.step()

        if self.red_sniper.action == self.blue_scout.position:
            blue_scout_dead = True
            logger.info("The Red Sniper killed the Blue Scout.")

            self.red_spy.post_step(np.array([int(blue_scout_dead), self.blue_scout.position]), 1.0, True)
            self.red_sniper.post_step(np.array([int(blue_scout_dead), self.red_spy.action]), 1.0, True)

        else:
            blue_scout_dead = False
            logger.info("The Red Sniper didn't kill the Blue Scout.")

            self.red_spy.post_step(np.array([int(blue_scout_dead), self.blue_scout.position]), -1.0, True)
            self.red_sniper.post_step(np.array([int(blue_scout_dead), self.blue_scout.position]), -1.0, True)

        logger.info("We're ending turn '{}'.".format(self.num_steps + 1))
        self.num_steps += 1

    def set_blue_scout_position(self):
        available_positions = self.num_gray_scouts + 1
        self.blue_scout.position = np.random.randint(available_positions)
        logger.info("Blue Scout position is: {}".format(self.blue_scout.position))


def main():
    model = SpyVsSpy(4)

    num_steps = 50

    for _ in range(num_steps):
        model.step()

    model.red_spy.drl_agent.print_weights()
    model.red_sniper.drl_agent.print_weights()


if __name__ == "__main__":
    main()
