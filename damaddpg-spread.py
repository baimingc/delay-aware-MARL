import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer_da import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
import copy
USE_CUDA = True  # torch.cuda.is_available()

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    
    

    delay_step = config.delay_step
    maddpg = MADDPG.init_from_env_with_delay(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim,
                                  delay_step = delay_step)
    
#     replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
#                                  [obsp.shape[0] + delay_step*2 for obsp in env.observation_space[:2]],
#                                  [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
#                                   for acsp in env.action_space[:2]])
    print(env.action_space, env.observation_space)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] + delay_step*5 for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='gpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        
#         zero_agent_actions = [np.array([0.0, 0.0]) for _ in range(maddpg.nagents-1)]
        zero_agent_actions = [np.array([0.0, 0.0, 0.0, 0.0, 0.0]) for _ in range(maddpg.nagents)]
        last_agent_actions = [zero_agent_actions for _ in range(delay_step)]
#         print(obs)
        obs = obs.tolist()
        for a_i, agent_obs in enumerate(obs[0]):
            for _ in range(len(last_agent_actions)):
                obs[0][a_i] = np.append(obs[0][a_i], last_agent_actions[_][a_i])
#                 obs[0][a_i] = np.append(agent_obs, last_agent_actions[_][a_i])
        obs = np.array(obs)
#         print(obs)
    
        for et_i in range(config.episode_length):
#             print(obs)

#                     agent_obs = np.append(agent_obs, last_agent_actions[_][a_i])
#             print(np.concatenate(obs[0], np.array(last_agent_actions).T))
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # augment the obs
            # get actions as torch Variables
#             print(torch_obs)
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            
#             print(torch_agent_actions)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
#             print('1', agent_actions)
    #        actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            # rearrange actions to be per environment
            if delay_step == 0:
                actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            else:
                agent_actions_tmp = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)][0][:]
#                 print('2', agent_actions_tmp)
                actions = last_agent_actions[0]
#                 print('3', actions)
#                 actions.append(agent_actions_tmp[-1])
#                 print('4', actions)
                last_agent_actions = last_agent_actions[1:]
#                 print('3', last_agent_actions)
                last_agent_actions.append(agent_actions_tmp)
#                 print('4', last_agent_actions)
#                 print('5', actions)
            actions = [actions]
#             print('1',actions)
            next_obs, rewards, dones, infos = env.step(copy.deepcopy(actions))
#             print('6', actions)
#             else:
#                 agent_actions_tmp = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)][0][:]
# #                 print('2', agent_actions_tmp)
#                 actions = last_agent_actions[0]
# #                 print('3', actions)
# #                 actions.append(agent_actions_tmp[-1])
# #                 print('4', actions)
#                 last_agent_actions = last_agent_actions[1:]
# #                 print('3', last_agent_actions)
#                 last_agent_actions.append(agent_actions_tmp)
# #                 print('4', last_agent_actions)
# #                 print('5', actions)
#             actions = [actions]
#             next_obs, rewards, dones, infos = env.step(copy.deepcopy(actions))
# #             print('6', actions)
            next_obs = next_obs.tolist()
            for a_i, agent_obs in enumerate(next_obs[0]):
                for _ in range(len(last_agent_actions)):
                    next_obs[0][a_i] = np.append(next_obs[0][a_i], last_agent_actions[_][a_i])
            next_obs = np.array(next_obs)
#                     next_obs[0][a_i] = np.append(agent_obs, 3*last_agent_actions[_][a_i])
#             print('3', agent_actions)

#             agent_actions[0] = agent_actions[0]*3
#             agent_actions[1] = agent_actions[1]*3
#             agent_actions[2] = agent_actions[2]/4

#             print('2',agent_actions)
#             print('4', obs)
#             print('5', next_obs)
#             print('1',agent_actions)
#             print(obs, next_obs)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents): #do not update the runner
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
#                     maddpg.update_adversaries()
                maddpg.prep_rollouts(device='gpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            # logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
            logger.add_scalars('agent%i/mean_episode_rewards' % a_i, {'reward': a_ep_rew}, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')
#             logger.export_scalars_to_json(str(log_dir / 'summary.json'))

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
#     parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e7), type=int)
    parser.add_argument("--n_episodes", default=10000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')
    parser.add_argument("--delay_step",
                    default = 1, type=int)

    config = parser.parse_args()

    run(config)
