import pkg_resources
import vrepgym

# The parameters that vrep_base env and other vrep environment might need
# Notice this dict is used by baseline.common.cmd_utils.make_vec_envs
# Hardcoded the arguments here instead of using argparse.parse_unknown_args because
# The unknown args are fed to baseline.run and used for policy network building.
vrep_full_arguments_dict = ['vrep_path','frame_skip','timestep_limit','obs_type','state_type','headless',
'random_start','server_silent','scene_path','log','terminal_penalty','reward_func']

def make_vrep(env_id='vrepgym.vrep_motor_env:VREPMotorEnv',max_episode_steps=None,**kwargs):
    '''
    The function wrapps vrep environment into forms of a gym environment.
    Params:env_id           : package and class name
    Params:max_episode_steps: maximum timesteps allowed in the environment
    Params:**kwargs         : All other parameters that VREPBaseEnv and env_id going to need
    '''
    assert ':' in env_id,"input 'path:class_name' that point to the environment"
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(env_id))
    cls = entry_point.load(False)
    print('making env',kwargs)
    env = cls(**kwargs)
    if max_episode_steps!=None:
        print('max_time_limit',max_episode_steps)
        from gym.wrappers.time_limit import TimeLimit
        env = TimeLimit(env,max_episode_steps=max_episode_steps)

    if not hasattr(env,'step'):
        env.step = env._step
        env.reset = env._reset
        env.seed = env._seed
		
        def render(mode):
            return env._render(mode,close=False)
        def close():
            return env._render('human',close=True)
        env.render = render
        env.close = close

    return env

def main():
    import gym
    import numpy as np
    env = make_vrep('vrepgym.vrep_motor_env:VREPMotorEnv')
    step = 0
    env.reset()
    while True:
        step+=1
        print(step)
        a = np.array([0] * 4)
        o, r, terminal, _ = env.step(a)
        if terminal:
            break
    env.close()
if __name__=='__main__':
    main()
