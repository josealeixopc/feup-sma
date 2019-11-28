from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid

from pade.acl.messages import AID, ACLMessage

import matplotlib.pyplot as plt
import numpy as np

import typing

## Rulebook for defining rules: https://cdn.1j1ju.com/medias/af/08/f4-the-werewolves-of-millers-hollow-rulebook.pdf
# Using the Basic Card Mix variant for 8 Players, with 2 Werewolves, 1 Fortune Teller and 5 Townfolks

class Player(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        self.talk()

    def talk(self):
        print("Hi! I'm a {} with ID '{}'.".format(type(self), self.unique_id))

class Werewolf(Player):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class Townfolk(Player):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class FortuneTeller(Townfolk):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class WerewolvesBasicMix(Model):
    def __init__(self, num_werewolves, num_townfolks, num_fortune_tellers):
        self.num_werewolves = num_werewolves
        self.num_townfolks = num_townfolks
        self.num_fortune_tellers = num_fortune_tellers

        self.schedule = RandomActivation(self)

        self.num_agents = 0
        self.num_steps = 0

        for _ in range(num_werewolves):
            self.add_player(Werewolf)
        
        for _ in range(num_townfolks):
            self.add_player(Townfolk)
        
        for _ in range(num_fortune_tellers):
            self.add_player(FortuneTeller)

    def add_player(self, Player_Type: typing.Type[Player]):
        assert issubclass(Player_Type, Player)

        a = Player_Type(self.num_agents, self)
        self.schedule.add(a)

        self.num_agents += 1

    def step(self):
        print("We're starting turn '{}'.".format(self.num_steps + 1))
        self.schedule.step()
        print("We're ending turn '{}'.".format(self.num_steps + 1))
        self.num_steps += 1


def main():
    model = WerewolvesBasicMix(2, 5, 1)

    num_steps = 10

    for _ in range(num_steps):
        model.step()

if __name__ == "__main__":
    main()