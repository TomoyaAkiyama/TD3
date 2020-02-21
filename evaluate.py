import os

import torch
import numpy as np
import gym

from td3 import TD3


def main(env_name, seed, hyper_params, eval_episodes=10):
    env = gym.make(env_name)

    state_dim = sum(list(env.observation_space.shape))
    action_dim = sum(list(env.action_space.shape))
    action_max = float(env.action_space.high[0])

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        'policy_freq': hyper_params['policy_freq']
    }

    agent = TD3(**kwargs)

    file_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(
        file_dir,
        'results',
        env_name,
        'seed' + str(seed),
        'learned_model'
    )
    agent.load(save_dir)

    env.seed(seed + 100)

    episode_rewards = []
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        sum_rewards = 0
        while not done:
            env.render()
            action = agent.rollout_actor.deterministic_action(state)
            next_state, reward, done, _ = env.step(action)
            sum_rewards += reward
            state = next_state
        episode_rewards.append(sum_rewards)
        print(f'Episode: {len(episode_rewards)} Sum Rewards: {sum_rewards:.3f}')

    avg_reward = np.mean(episode_rewards)
    print('\n---------------------------------------')
    print(f'Evaluation over {eval_episodes} episodes: {avg_reward:.3f}')
    print('---------------------------------------')


if __name__ == '__main__':
    env_name = 'Swimmer-v2'
    seed = 2

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

    main(env_name, seed, hyper_parameters)
