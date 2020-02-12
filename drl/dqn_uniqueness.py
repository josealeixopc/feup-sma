# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque

import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        state_to_num_actions_taken = {}
        for i in self.state_size:
            state_to_num_actions_taken[i] = [0] * self.action_size

        for state, action, _, _, _ in minibatch:
            state_to_num_actions_taken[state][action] += 1  # this action was called in this state

        uniqueness_average = DQNAgent.calculate_uniqueness_average(state_to_num_actions_taken)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + uniqueness_average + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def print_weights(self):
        for layer in self.model.layers:
            print(layer.get_config(), layer.get_weights())

    @staticmethod
    def calculate_uniqueness_average(state_to_num_actions_taken: dict):
        max_uniqueness_values = []

        for key in state_to_num_actions_taken:
            max_num_times_action_taken_for_obs = max(state_to_num_actions_taken[key])
            num_times_action_taken = sum(state_to_num_actions_taken[key])

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
