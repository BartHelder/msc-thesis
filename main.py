from train import train
import torch

# Some parameters
ac_params = {'col':
                {'control_channel': 'col',
                 'discount_factor': 0.9,
                 'n_hidden_actor': 8,
                 'nn_stdev_actor': 0.01,
                 'learning_rate_actor': 0.1,
                 'action_scaling': None,
                 'n_hidden_critic': 8,
                 'nn_stdev_critic': 0.01,
                 'learning_rate_critic': 0.1,
                 'tau_target_critic': 0.01,
                 'tracked_state': 11,
                 'ac_states': [2],
                 'reward_weight': 0.1},
             'lon':
                {'control_channel': 'lon',
                 'discount_factor': 0.9,
                 'n_hidden_actor': 8,
                 'nn_stdev_actor': 0.1,
                 'learning_rate_actor': 5,
                 'action_scaling': None,
                 'n_hidden_critic': 8,
                 'nn_stdev_critic': 0.1,
                 'learning_rate_critic': 5,
                 'tau_target_critic': 0.01,
                 'tracked_state': 7,
                 'ac_states': [4]}}

rls_params = {'state_size': 15,
              'action_size': 2,
              'gamma': 1,
              'covariance': 10**8,
              'constant': False}

env_params = {'initial_velocity': 20,
              'initial_flight_path_angle': 0,
              'initial_altitude': 0,
              'dt': 0.01,
              't_max': 120,
              't_switch': 60}
env_params.update({'step_switch': int(env_params['t_switch'] / env_params['dt']),
                   'n_steps': int(env_params['t_max'] / env_params['dt'])})


env_params['step_switch'] = int(env_params['t_switch'] / env_params['dt'])
env_params['n_steps'] = int(env_params['t_max'] / env_params['dt'])

path = "results/mar/4/1/"
training_logs = train(env_params=env_params, ac_params=ac_params, rls_params=rls_params, path=path, seed=2)
