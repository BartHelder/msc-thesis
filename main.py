# Standard library
import itertools
import time
import os

# Other
import torch
import numpy as np
import seaborn as sns

# Custom made
from agents import DHPAgent
from model import RecursiveLeastSquares
from heli_models import Helicopter6DOF
from PID import LatPedPID, CollectivePID6DOF
from util import Logger, get_ref, envelope_limits_reached, plot_rls_weights, plot_neural_network_weights, plot_stats





# Some parameters
agent_params = {'col':
                    {'control_channel': 'col',
                     'discount_factor': 0.95,
                     'n_hidden_actor': 8,
                     'nn_stdev_actor': 0.05,
                     'learning_rate_actor': 0.1,
                     'action_scaling': None,
                     'n_hidden_critic': 8,
                     'nn_stdev_critic': 0.05,
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
                     'learning_rate_actor': 8,
                     'action_scaling': None,
                     'n_hidden_critic': 8,
                     'nn_stdev_critic': 0.1,
                     'learning_rate_critic': 8,
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




