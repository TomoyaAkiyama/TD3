import os
import time

import gym
import numpy as np
import torch

from td3 import TD3
from replay_buffer import ReplayBuffer

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def eval_policy(policy, env_name, seed, eval_episodes=10):
    env = gym.make(env_name)
    env.seed(seed + 100)

    episode_rewards = []
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        sum_rewards = 0
        while not done:
            # env.render()
            action = policy.deterministic_action(state)
            next_state, reward, done, _ = env.step(action)
            sum_rewards += reward
            state = next_state
        episode_rewards.append(sum_rewards)

    avg_reward = np.mean(episode_rewards)
    print('---------------------------------------')
    print(f'Evaluation over {eval_episodes} episodes: {avg_reward:.3f}')
    print('---------------------------------------')

    return avg_reward


def main(env_name, seed, hyper_params):
    env = gym.make(env_name)

    state_dim = sum(list(env.observation_space.shape))
    action_dim = sum(list(env.action_space.shape))
    action_max = float(env.action_space.high[0])

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    device = torch.device('cuda' if hyper_params['use_cuda'] else 'cpu')

    kwargs = {
        'device': device,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'action_max': action_max,
        'gamma': hyper_params['gamma'],
        'tau': hyper_params['tau'],
        'lr': hyper_params['lr'],
        'policy_noise': hyper_params['policy_noise'] * action_max,
        'noise_clip': hyper_params['noise_clip'] * action_max,
        'exploration_noise': hyper_params['exploration_noise'] * action_max,
        'policy_freq': hyper_params['policy_freq']
    }

    agent = TD3(**kwargs)
    replay_buffer = ReplayBuffer(state_dim, action_dim, device, max_size=int(1e6))

    file_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(
        file_dir,
        'results',
        env_name,
        'seed' + str(seed)
    )
    os.makedirs(save_dir, exist_ok=True)

    evals = [eval_policy(agent.rollout_actor, env_name, seed)]

    state = env.reset()
    episode_reward = 0
    episode_time_step = 0
    episode_num = 0

    episode_start = time.time()
    for t in range(hyper_params['max_time_step']):
        episode_time_step += 1

        if t < hyper_params['initial_time_step']:
            action = env.action_space.sample()
        else:
            action = agent.rollout_actor.select_action(state)

        next_state, reward, done, _ = env.step(action)
        done_buffer = done if episode_time_step < env._max_episode_steps else False
        replay_buffer.add(state, next_state, action, reward, done_buffer)

        state = next_state
        episode_reward += reward

        if t >= hyper_params['initial_time_step']:
            agent.train(replay_buffer, batch_size=hyper_params['batch_size'])

        if done:
            print(f'Total T: {t + 1} Episode Num: {episode_num + 1} Reward: {episode_reward:.3f}',
                  f'(Frame/sec {episode_time_step / (time.time() - episode_start):.3f})')
            # Reset environment
            state = env.reset()
            episode_reward = 0
            episode_time_step = 0
            episode_num += 1
            episode_start = time.time()

        if (t + 1) % hyper_params['eval_freq'] == 0:
            test_start = time.time()
            # test policy
            evals.append(eval_policy(agent.rollout_actor, env_name, seed))
            test_time = time.time() - test_start
            episode_start += test_time

    evals = np.array(evals)
    np.savetxt(os.path.join(save_dir, 'Episode_Rewards.txt'), evals)
    plt.figure()
    time_step = np.arange(len(evals)) * hyper_params['eval_freq']
    plt.plot(time_step, evals)
    plt.xlabel('Time Steps')
    plt.ylabel('Episode Rewards')
    plt.grid()
    file_name = 'Episode_Rewards.png'
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    plt.close()

    model_path = os.path.join(save_dir, 'learned_model')
    os.makedirs(model_path, exist_ok=True)
    agent.save(model_path)


if __name__ == '__main__':
    env_name = 'HalfCheetah-v2'
    seed = 1

    hyper_parameters = {
        'max_time_step': 1000000,
        'initial_time_step': 10000,
        'batch_size': 256,
        'eval_freq': 5000,
        'gamma': 0.99,
        'tau': 0.005,
        'lr': 3e-4,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'exploration_noise': 0.1,
        'policy_freq': 2,
        'use_cuda': torch.cuda.is_available(),
    }

    start = time.time()
    main(env_name, seed, hyper_parameters)
    print('\nelapsed time : {}'.format(time.time() - start))


