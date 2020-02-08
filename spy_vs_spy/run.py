import argparse
import os
from datetime import datetime

from spy_vs_spy.env.spy_vs_spy_ma_env import RedSpyEnv, RedSniperEnv
from spy_vs_spy import utils

AVAILABLE_ENVIRONMENTS = ['red-spy-env', 'red-sniper-env']
AVAILABLE_ALGORITHMS = ['dqn']

DEFAULT_TRAINING_TIMESTEPS = 10000
TENSORBOARD_DIR_NAME = 'tensorboard'
TRAININGS_DIR_NAME = 'trainings'


def _get_gym_environment(environment):
    if environment == AVAILABLE_ENVIRONMENTS[0]:
        env = RedSpyEnv('red-spy', 'localhost:50051')
    elif environment == AVAILABLE_ENVIRONMENTS[1]:
        env = RedSniperEnv('red-sniper', 'localhost:50051')
    else:
        raise Exception("Environment '{}' is unknown.".format(environment))

    return env


def _load_model(algorithm, model_path):
    from stable_baselines import DQN

    if algorithm == AVAILABLE_ALGORITHMS[0]:
        model = DQN.load(model_path)
    else:
        raise Exception("Algorithm '{}' is unknown.".format(algorithm))


def observe(environment, algorithm, model_path, num_steps):
    env = _get_gym_environment(environment)
    model = _load_model(algorithm, model_path)

    # Enjoy trained agent
    obs = env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()


def evaluate(environment, algorithm, model_path, n_eval_episodes=100):
    from stable_baselines.common.evaluation import evaluate_policy

    env = _get_gym_environment(environment)
    model = _load_model(algorithm, model_path)

    # Evaluate the agent
    mean_reward, n_steps = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)

    print("Evaluated {} for {} episodes. Mean reward: {}. N_Steps: {}."
          .format(model_path, n_eval_episodes, mean_reward, n_steps))


def train(environment, algorithm, timesteps):
    from stable_baselines import DQN
    from stable_baselines.bench import Monitor

    now = datetime.now()
    starting_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    training_info_dir = "trainings" + os.path.sep
    current_training_info = "{}-{}-{}".format(starting_time, algorithm, environment)
    current_training_info_dir = training_info_dir + current_training_info + os.path.sep

    model_file_path = current_training_info_dir + "model"
    log_file_path = current_training_info_dir + "monitor.csv"

    tensorboard_dir = training_info_dir + TENSORBOARD_DIR_NAME + os.path.sep

    dirs_to_create = [model_file_path, tensorboard_dir, model_file_path]

    for directory in dirs_to_create:
        utils.create_dir(directory)

    env = _get_gym_environment(environment)

    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    env = Monitor(env, filename=log_file_path, allow_early_resets=True)

    if algorithm == AVAILABLE_ALGORITHMS[0]:
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_dir)
    else:
        raise Exception("Algorithm '{}' is unknown.".format(algorithm))

    # Train the agent
    model.learn(total_timesteps=timesteps, tb_log_name=current_training_info)

    model.save(model_file_path)

    print("Finished training model: {}. Saved training info in: {}".format(model, current_training_info_dir))


def check_arguments(args):
    if args.task not in ['train', 'eval', 'observe']:
        raise argparse.ArgumentError("Only allow [train|eval|observe].")

    if args.environment not in AVAILABLE_ENVIRONMENTS:
        raise argparse.ArgumentError(
            "Environment '{}' does not belong to available environments ({}).".format(args.environment,
                                                                                      AVAILABLE_ENVIRONMENTS))

    if args.algorithm not in AVAILABLE_ALGORITHMS:
        raise argparse.ArgumentError(
            "Algorithm '{}' does not belong to available algorithms ({}).".format(args.algorithm, AVAILABLE_ALGORITHMS))

    if (args.timesteps <= 0) or (not isinstance(args.timesteps, int)):
        raise argparse.ArgumentTypeError("Number of timesteps must be a positive integer and different than zero.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a DRL model in a SpyVsSpy env.')
    parser.add_argument('task', help='train|eval|observe')
    parser.add_argument('environment',
                        help='The environment to run. One of: '.format(AVAILABLE_ENVIRONMENTS))
    parser.add_argument('algorithm', help='The DRL algorithm. One of: '.format(AVAILABLE_ALGORITHMS))
    parser.add_argument('--timesteps', type=int, default=DEFAULT_TRAINING_TIMESTEPS,
                        help='Number of timesteps (default: {})'.format(DEFAULT_TRAINING_TIMESTEPS))
    parser.add_argument('--model', type=str, help='Path of model.')

    args = parser.parse_args()

    check_arguments(args)

    if args.task == 'train':
        train(args.environment, args.algorithm, args.timesteps)
    elif args.task == 'eval':
        evaluate(args.environment, args.algorithm, args.model)
    elif args.task == 'observe':
        observe(args.environment, args.algorithm, args.model, args.timesteps)
