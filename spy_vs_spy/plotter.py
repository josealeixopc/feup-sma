import matplotlib.pyplot as plt
from stable_baselines import results_plotter


if __name__ == '__main__':
    results_plotter.plot_results(["/tmp/gym/red-sniper"], 10e6, results_plotter.X_TIMESTEPS, "Breakout")
    plt.show()
