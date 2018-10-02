"""
Helpers for scripts like run_atari.py.
"""

import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.retro_wrappers import RewardScaler
import time

def make_vec_env(env_id, env_type, num_env, seed, wrapper_kwargs=None, start_index=0, reward_scale=1.0,env_args=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            if env_type == 'atari':
                env = make_atari(env_id)
            elif env_type == 'vrep':
                from vrepgym import make_vrep                
                vrep_args_dict = parse_unknown_args(env_args,make_vrep.vrep_full_arguments_dict)
                timestep_limit = vrep_args_dict['timestep_limit'] if 'timestep_limit' in vrep_args_dict.keys() else None  
                logger.log(str(env_id.split(':')[-1])+'is constructed with parameters as such:')
                logger.log(vrep_args_dict)
                env = make_vrep.make_vrep(env_id,remote_port=rank+20000,max_episode_steps=timestep_limit,**vrep_args_dict)
            else:
                env = gym.make(env_id)
            env.seed(seed + 10000*mpi_rank + rank if seed is not None else None)
            env = Monitor(env,
                          logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(rank)),
                          allow_early_resets=True)

            if env_type == 'atari': return wrap_deepmind(env, **wrapper_kwargs)
            elif reward_scale != 1: return RewardScaler(env, reward_scale)
            else: return env
        return _thunk
    set_global_seeds(seed)
    if num_env > 1: return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else: return DummyVecEnv([make_env(start_index)])

def make_mujoco_env(env_id, seed, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed  + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def mujoco_arg_parser():
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
	# vrep_base environment related arguments
    parser.add_argument('--vrep_path',help='absolute path of vrep',type=str,default='/home/elessar/reinforcement/vrep_baselines/V-REP_PRO_EDU_V3_3_2_64_Linux')
    parser.add_argument('--frame_skip',type=int,default=1)
    parser.add_argument('--timestep_limit',type=int,default=500)
    parser.add_argument('--obs_type',type=str,default='state')
    parser.add_argument('--state_type',type=str,default='world')
    parser.add_argument('--headless',default=False,action='store_true')
    parser.add_argument('--random_start',default=False,action='store_true')
    parser.add_argument('--server_silent',default=False,action='store_true')
    # vrep_env environment related arguments
    parser.add_argument('--env', help='environment class', type=str, default='vrepgym.vrep_motor_env:VREPMotorEnv')
    parser.add_argument('--scene_path',help='environement scene path',type=str,default='motor_control.ttt')
    parser.add_argument('--log',help='log into console and files',default=False,action='store_true')
    # parser.add_argument('--reward_baseline',type=float,default=2.95)
    parser.add_argument('--terminal_penalty',type=float,default=0)
    parser.add_argument('--reward_func',type=float,default=1)
    # algorithm and network related arguments
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='mlp')
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    return parser

def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser


def parse_unknown_args(args,known_args=None):
    """
    Parse arguments not consumed by arg parser into a dictionary
    Params: args: can be of one of the following two cases
                  1. list containing args that are not in the common arg_parser.
                     These arguments are for ppo2 and network structure
                  2. NameSpace containing all args passed from common arg_parser
                     Environment related arguments are filted out, kept and parsed.
    Params: known_args: environment related arguments
    """
    arg_list = args
    if not isinstance(arg_list,list):
        arg_list = ['--'+str(k)+'='+str(getattr(arg_list,k)) for k in vars(arg_list) if (str(k) in known_args)]
    retval = {}
    preceded_by_key = False
    
    for arg in arg_list:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key]=value
            else:
                key = arg[2:]
                preceded_by_key = True            
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False
    if known_args !=None:
        for k in retval.keys():
            try:
                retval[k]=eval(retval[k])
            except:
                retval[k]=retval[k]
    return retval
