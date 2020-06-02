ac_params_train = {'col':
                {'control_channel': 'col',
                 'discount_factor': 0.99,
                 'n_hidden_actor': 10,
                 'nn_stdev_actor': 0.1,
                 'learning_rate_actor': 0.01,
                 'action_scaling': None,
                 'n_hidden_critic': 10,
                 'nn_stdev_critic': 0.1,
                 'learning_rate_critic': 0.01,
                 'tau_target_critic': 1,
                 'tracked_state': 11,
                 'ac_states': [2],
                 'reward_weight': 0.1},
             'lon':
                {'control_channel': 'lon',
                 'discount_factor': 0.95,
                 'n_hidden_actor': 10,
                 'nn_stdev_actor': 0.1,
                 'learning_rate_actor': 5,
                 'action_scaling': None,
                 'n_hidden_critic': 10,
                 'nn_stdev_critic': 0.1,
                 'learning_rate_critic': 5,
                 'tau_target_critic': 1,
                 'tracked_state': 7,
                 'ac_states': [4]}}

ac_params_test = {
    'col': {
        'control_channel': 'col',
        'discount_factor': 0.99,
        'n_hidden_actor': 10,
        'nn_stdev_actor': 0.1,
        'learning_rate_actor': 0.05,
        'action_scaling': None,
        'n_hidden_critic': 10,
        'nn_stdev_critic': 0.1,
        'learning_rate_critic': 0.05,
        'tau_target_critic': 1,
        'tracked_state': 11,
        'ac_states': [2],
        'reward_weight': 0.1},
    'lon': {
        'control_channel': 'lon',
        'discount_factor': 0.95,
        'n_hidden_actor': 10,
        'nn_stdev_actor': 0.1,
        'learning_rate_actor': 0.05,
        'action_scaling': None,
        'n_hidden_critic': 10,
        'nn_stdev_critic': 0.1,
        'learning_rate_critic': 0.05,
        'tau_target_critic': 1,
        'tracked_state': 7,
        'ac_states': [4]}
}

rls_params = {'state_size': 15,
              'action_size': 2,
              'gamma': 1,
              'covariance': 10**8,
              'constant': False}

pid_params = {"Ky": 0.03,
              "Ky_int": 0.0025,
              "Ky_dot": -0.08,
              "Kphi": 5,
              "Kphi_int": 2.5,
              "Kp": -2,
              "Kpsi": 5,
              "Kpsi_int": 2,
              "Kr": -2,
              "Kh": 0.005}

env_params_train = {
    'initial_velocity': 15,
    'initial_flight_path_angle': 0,
    'initial_altitude': 0,
    'dt': 0.01,
    't_max': 120,
    't_switch': 60}
env_params_train['step_switch'] = int(env_params_train['t_switch'] / env_params_train['dt'])
env_params_train['n_steps'] = int(env_params_train['t_max'] / env_params_train['dt'])

env_params_test1 = {
    'initial_velocity': 0,
    'initial_flight_path_angle': 0,
    'initial_altitude': 0,
    'dt': 0.01,
    't_max': 40,
    't_switch': 60}
env_params_test1['step_switch'] = int(env_params_test1['t_switch'] / env_params_test1['dt'])
env_params_test1['n_steps'] = int(env_params_test1['t_max'] / env_params_test1['dt'])

env_params_test2 = {'initial_velocity': 18,
                   'initial_flight_path_angle': -6,
                   'initial_altitude': 30,
                   'dt': 0.01,
                   't_max': 30,
                   't_switch': 30}
env_params_test2['step_switch'] = int(env_params_test2['t_switch'] / env_params_test2['dt'])
env_params_test2['n_steps'] = int(env_params_test2['t_max'] / env_params_test2['dt'])
