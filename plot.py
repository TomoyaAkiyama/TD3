import os

import matplotlib.pyplot as plt
import numpy as np


def episode_profits_plot(env_name, seed, hyper_params, x_min, x_max, y_min, y_max):
    eval_freq = hyper_params['eval_freq']

    file_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(
        file_dir,
        'results',
        env_name,
        f'seed{seed}'
    )
    file_name = os.path.join(save_dir, 'Episode_Rewards.txt')
    episode_profits = np.loadtxt(file_name)

    time_steps = np.arange(len(episode_profits)) * eval_freq

    plt.figure()
    plt.plot(time_steps, episode_profits)
    plt.grid()
    plt.xlabel('Time Step')
    plt.ylabel('Episode Rewards')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    file_path = os.path.join(save_dir, 'Episode_Rewards.png')
    plt.savefig(file_path)


def average_plot(env_name, seeds, hyper_params, x_min, x_max, y_min, y_max):
    eval_freq = hyper_params['eval_freq']

    episode_profits_list = []
    file_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(
        file_dir,
        'results',
        env_name,
    )
    for seed in seeds:
        file_name = os.path.join(save_dir, f'seed{seed}', 'Episode_Rewards.txt')
        episode_profits_list.append(np.loadtxt(file_name))

    avg = np.mean(episode_profits_list, axis=0)
    std = np.std(episode_profits_list, axis=0)
    time_steps = np.arange(len(avg)) * eval_freq

    plt.figure()
    plt.plot(time_steps, avg)
    plt.fill_between(time_steps, avg - std, avg + std, facecolor='c', alpha=0.5)
    plt.grid()
    plt.xlabel('Time Step')
    plt.ylabel('Average Rewards')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    file_path = os.path.join(save_dir, 'Average_Rewards.png')
    plt.savefig(file_path)


if __name__ == '__main__':
    env_name = 'HalfCheetah-v2'

    hyper_parameters = {
        'max_time_step': 1000000,
        'initial_time_step': 10000,
        'batch_size': 256,
        'exploration_noise': 0.1,
        'eval_freq': 5000,
        'gamma': 0.99,
        'tau': 0.005,
        'lr': 3e-4,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'policy_freq': 2
    }

    seeds = range(1, 11)
    x_min = 0
    x_max = 1000000
    y_min = -500
    y_max = 10000

    for seed in seeds:
        episode_profits_plot(env_name, seed, hyper_parameters, x_min, x_max, y_min, y_max)
    average_plot(env_name, seeds, hyper_parameters, x_min, x_max, y_min, y_max)
