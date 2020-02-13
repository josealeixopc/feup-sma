import sys

import matplotlib.pyplot as plt
from stable_baselines import results_plotter

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import json

Y_REWARDS = 'r'
Y_EPISODE_LENGTH = 'l'
Y_TIME_ELAPSED = 't'

lines = ["-", "--", ":", "-."]
markers = ['o', 'x', '+', '^']
colors = ['#000000', '#222222', '#444444', '#666666']


def smooth_moving_average(x, y, window_size):
    # Understanding convolution with window size: https://stackoverflow.com/a/20036959/7308982
    y_new = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')

    # We need to trim the last values, because the "valid" mode returns a list with size max(M, N) - min(M, N) + 1.
    # See here: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html
    x_trimmed = x[:-(window_size - 1)]

    return x_trimmed, y_new


if __name__ == '__main__':
    # results_plotter.plot_results(["/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-08-23-25-53-dqn-red-sniper-env"], 10e6, results_plotter.X_TIMESTEPS, "Breakout")
    # plt.show()
    # sys.exit(0)

    keras_dqn_uniqueness_file = "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-13-19-14-28-dqn-keras-uniqueness-red-sniper-env/monitor.csv/openaigym.episode_batch.0.32515.stats.json"
    keras_dqn_file = "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-13-14-25-00-dqn-keras-red-sniper-env/monitor.csv/openaigym.episode_batch.0.16963.stats.json"

    with open(keras_dqn_uniqueness_file) as f:
        keras_dqn_uniqueness_json = json.load(f)

    with open(keras_dqn_file) as f:
        keras_dqn_json = json.load(f)

    all_rewards = keras_dqn_uniqueness_json['episode_rewards'] + keras_dqn_json[
        'episode_rewards']
    all_lengths = keras_dqn_uniqueness_json['episode_lengths'] + keras_dqn_json[
        'episode_lengths']

    plt.style.use('ggplot')

    handles = []

    x, y = results_plotter.window_func(np.array(range(len(keras_dqn_json['episode_rewards']))),
                                       np.array(keras_dqn_json['episode_rewards']),
                                       100,
                                       np.mean)

    line, = plt.plot(x, y, linewidth=1, label='dqn')
    handles.append(line)

    # x, y = smooth_moving_average(range(len(keras_dqn_uniqueness_json['episode_rewards'])),
    #                              keras_dqn_uniqueness_json['episode_rewards'],
    #                              100)

    x, y = results_plotter.window_func(np.array(range(len(keras_dqn_json['episode_rewards']))),
                                       np.array(keras_dqn_uniqueness_json['episode_rewards']),
                                       100,
                                       np.mean)

    line, = plt.plot(x, y, linewidth=1, label='dqn-uniqueness')
    handles.append(line)

    plt.xlim(left=0)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Rewards per Episode")
    plt.tight_layout()
    plt.legend(handles=handles)
    ax = plt.gca()
    ratio = 0.4
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    # plt.savefig("rewards.eps", format='eps', bbox_inches='tight')
    plt.show()
