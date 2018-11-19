import argparse
import time
import os
import logging
from copy import copy
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
import vrepgym
import vrepgym.make_vrep as make_vrep

import gym
import tensorflow as tf
from mpi4py import MPI

def run(env_args,env_id, seed, noise_type, layer_norm, evaluation, hidden_unit, layer_num,explore_ratio=1/2,**kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    
    # Create envs.
    if env_id.split('.')[0]!='vrepgym':
        env = gym.make(env_id)
    else:
        remote_port = 20000
        env = make_vrep.make_vrep(env_id,remote_port=remote_port,**env_args)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    if evaluation and rank==0:
        if env_id.split('.')[0]!='vrepgym':
            eval_env = gym.make(env_id)
        else:
            eval_env = make_vrep.make_vrep(env_id,remote_port=remote_port+1,**env_args)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        # env = bench.Monitor(env, None) # why is there a double wrapp?
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            exp_timestep = int(explore_ratio*kwargs['nb_epochs']*kwargs['nb_epoch_cycles']*kwargs['nb_rollout_steps'])
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions),sigma=float(stddev)*np.ones(nb_actions),decay_period=exp_timestep)
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm,layer_num=layer_num,hidden_unit=hidden_unit)
    actor = Actor(nb_actions, layer_norm=layer_norm,layer_num=layer_num,hidden_unit=hidden_unit)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # The environment related arguments
    parser.add_argument('--vrep_path',help='absolute path of vrep',type=str,default='/home/elessar/reinforcement/vrep_baselines/V-REP_PRO_EDU_V3_3_2_64_Linux')
    parser.add_argument('--frame_skip',type=int,default=1)
    parser.add_argument('--timestep_limit',type=int,default=500)
    parser.add_argument('--obs_type',type=str,default='state')
    parser.add_argument('--state_type',type=str,default='world')
    parser.add_argument('--headless',default=False,action='store_true')
    parser.add_argument('--random_start',default=False,action='store_true')
    parser.add_argument('--server_silent',default=False,action='store_true')
    # vrep_env environment related arguments
    parser.add_argument('--scene_path',help='environement scene path',type=str,default='motor_control.ttt')
    parser.add_argument('--log',help='log into console and files',default=False,action='store_true')
    # parser.add_argument('--reward_baseline',type=float,default=2.95)
    parser.add_argument('--terminal_penalty',type=float,default=0)
    parser.add_argument('--reward_func',type=float,default=1)

    # The algorithm related arguments
    parser.add_argument('--env_id', help='environment class', type=str, default='vrepgym.vrep_motor_env:VREPMotorEnv')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--hidden_unit',help='number of cells in hidden unit',type=int,default=128)
    parser.add_argument('--layer_num',help='number of layers for policy/value function',type=int,default=2)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=100)  # with default settings, perform 2e5 steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20) # number of roll out times per epoch
    parser.add_argument('--nb-train-steps', type=int, default=500)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=500)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='normal_0.99') #'adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--explore_ratio',type=float,default=1,help='portion of timesteps applying annealing of action noise')
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    parser.add_argument('--load_path',type=str,default=None)
    args = parser.parse_args()
    # explore ratio has to be less than 1
    assert args.explore_ratio<=1
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    if args.env_id.split('.')[0]=='vrepgym':
        dict_args_copy = copy(dict_args)
        env_args = {k:dict_args_copy.pop(k) for k in dict_args.keys() if k in make_vrep.vrep_full_arguments_dict}
        return dict_args_copy,env_args
    else:
        return dict_args,None


if __name__ == '__main__':
    dict_args,env_args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    run(env_args,**dict_args)
