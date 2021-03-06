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

    # 5 scout files
    keras_dqn_uniqueness_files = [
        "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-13-18-08-30-dqn-keras-uniqueness-red-sniper-env/monitor.csv/openaigym.episode_batch.0.30768.stats.json",
        "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-13-19-14-28-dqn-keras-uniqueness-red-sniper-env/monitor.csv/openaigym.episode_batch.0.32515.stats.json",
        "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-13-21-14-36-dqn-keras-uniqueness-red-sniper-env/monitor.csv/openaigym.episode_batch.0.5046.stats.json"
        ]
    keras_dqn_files = [
        "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-13-14-25-00-dqn-keras-red-sniper-env/monitor.csv/openaigym.episode_batch.0.16963.stats.json",
        "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-13-20-18-50-dqn-keras-red-sniper-env/monitor.csv/openaigym.episode_batch.0.3661.stats.json",
        "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-13-21-48-28-dqn-keras-red-sniper-env/monitor.csv/openaigym.episode_batch.0.9769.stats.json"
    ]

    # 20 scout files
    # keras_dqn_uniqueness_files = [
    #     "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-14-00-29-38-dqn-keras-uniqueness-red-sniper-env/monitor.csv/openaigym.episode_batch.0.7257.stats.json",
    #     "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-14-04-46-27-dqn-keras-uniqueness-red-sniper-env/monitor.csv/openaigym.episode_batch.0.10363.stats.json",
    #     "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-14-11-42-41-dqn-keras-uniqueness-red-sniper-env/monitor.csv/openaigym.episode_batch.0.16056.stats.json"
    # ]
    #
    # keras_dqn_files = [
    #     "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-14-01-41-46-dqn-keras-red-sniper-env/monitor.csv/openaigym.episode_batch.0.8272.stats.json",
    #     "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-14-02-53-31-dqn-keras-red-sniper-env/monitor.csv/openaigym.episode_batch.0.9138.stats.json",
    #     "/home/jazz/Projects/FEUP/ProDEI/feup-sma/spy_vs_spy/trainings/2020-02-14-03-50-13-dqn-keras-red-sniper-env/monitor.csv/openaigym.episode_batch.0.9757.stats.json"
    # ]

    list_rewards_dqn_uniqueness = []

    for json_file in keras_dqn_uniqueness_files:
        with open(json_file) as f:
            keras_dqn_uniqueness_json = json.load(f)
            list_rewards_dqn_uniqueness.append(keras_dqn_uniqueness_json['episode_rewards'])

    list_rewards_dqn = []

    for json_file in keras_dqn_files:
        with open(json_file) as f:
            keras_dqn_json = json.load(f)
            list_rewards_dqn.append(keras_dqn_json['episode_rewards'])

    average_rewards_dqn_uniqueness = [sum(elem)/len(elem) for elem in zip(*list_rewards_dqn_uniqueness)]
    average_rewards_dqn = [sum(elem)/len(elem) for elem in zip(*list_rewards_dqn)]

    plt.style.use('ggplot')

    handles = []

    x, y = results_plotter.window_func(np.array(range(len(average_rewards_dqn))),
                                       np.array(average_rewards_dqn),
                                       200,
                                       np.mean)

    line, = plt.plot(x, y, linewidth=1, label='DQN')
    handles.append(line)

    # x, y = smooth_moving_average(range(len(keras_dqn_uniqueness_json['episode_rewards'])),
    #                              keras_dqn_uniqueness_json['episode_rewards'],
    #                              100)

    x, y = results_plotter.window_func(np.array(range(len(average_rewards_dqn_uniqueness))),
                                       np.array(average_rewards_dqn_uniqueness),
                                       200,
                                       np.mean)

    line, = plt.plot(x, y, linewidth=1, label='DQN with Coherence Index')
    handles.append(line)

    plt.xlim(left=0)
    plt.title("5 scouts")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Reward per Episode")
    plt.tight_layout()
    plt.legend(handles=handles)
    ax = plt.gca()
    ratio = 0.4
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    plt.savefig("rewards_5_scouts.eps", format='eps', bbox_inches='tight')
    plt.show()
