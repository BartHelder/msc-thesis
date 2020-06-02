import copy
import datetime
import time
import logging
from itertools import count
from collections import defaultdict

import numpy as np
import pandas as pd

import torch

from agents import DHPAgent
from model import RecursiveLeastSquares
from heli_models import Helicopter3DOF
from plotting import plot_stats_3dof
from PID import CollectivePID3DOF


t0 = time.time()
torch.manual_seed(2)
np.random.seed(2)

# Some parameters
agent_parameters = {'col':
                    {'control_channel': 'col',
                     'discount_factor': 0.9,
                     'n_hidden_actor': 10,
                     'nn_stdev_actor': 0.1,
                     'learning_rate_actor': 0.1,
                     'action_scaling': 5,
                     'n_hidden_critic': 10,
                     'nn_stdev_critic': 0.1,
                     'learning_rate_critic': 0.1,
                     'tau_target_critic': 0.01,
                     'tracked_state': 1,
                     'ac_states': [3],
                     'reward_weight': 0.01},
                    'lon':
                    {'control_channel': 'lon',
                     'discount_factor': 0.99,
                     'n_hidden_actor': 6,
                     'nn_stdev_actor': 0.2,
                     'learning_rate_actor': 0.2,
                     'action_scaling': 15,
                     'n_hidden_critic': 6,
                     'nn_stdev_critic': 0.2,
                     'learning_rate_critic': 0.2,
                     'tau_target_critic': 1,
                     'tracked_state': 4,
                     'ac_states': [5]}}
rls_parameters = {'state_size': 7,
                  'action_size': 2,
                  'gamma': 1,
                  'covariance': 10**4,
                  'constant': False}
V_INITIAL = 10
dt = 0.01
t_max = 120
n_steps = int(t_max / dt)

# EnvironmentSt
config_path = "/home/bart/PycharmProjects/msc-thesis/config_3dof.json"
env = Helicopter3DOF(dt=dt, t_max=t_max)
env.setup_from_config(task="sinusoid", config_path=config_path)
obs, trim_action = env.reset(v_initial=V_INITIAL)
ref = env.get_ref()

# incremental RLS estimator
RLS = RecursiveLeastSquares(**rls_parameters)

# Agents:
collective_pid = CollectivePID3DOF(h_ref = 5, dt=dt)
agent_col = DHPAgent(**agent_parameters['col'], action_network_final_layer='tanh')
agent_lon = DHPAgent(**agent_parameters['lon'], action_network_final_layer='tanh')
agents = [agent_col, agent_lon]

# Excitation signal for the RLS estimator
excitation = np.zeros((1000, 2))
for j in range(400):
    excitation[j, 1] = -np.sin(np.pi * j / 50)
    excitation[j + 400, 0] = np.sin(2 * np.pi * j / 50) * 2
excitation = np.deg2rad(excitation)

# Bookkeeping
excitation_phase = False
done = False
step = 0
rewards = np.zeros(2)
t_start = time.time()
update_col = False
update_lon = True

stats = []
network_sequence = ['a', 'c', 'tc']
layer_sequence = ['i', 'o']

weight_stats = {'t': [],
                'col':
                    {'nn': {
                        'a': {'i': [],
                              'o': []},
                        'c': {'i': [],
                              'o': []},
                        'tc': {'i': [],
                               'o': []}
                            },
                     'rls': {'F': [],
                             'G': []
                             }
                     },
                'lon': {
                    'nn': {
                        'a': {'i': [],
                              'o': []},
                        'c': {'i': [],
                              'o': []},
                        'tc': {'i': [],
                               'o': []}
                            },
                    'rls': {'F': [],
                            'G': []}
                     }
                }

#  Main loop
for step in range(n_steps):

    if step == 800:
        excitation_phase = False

    if step == 6000:
        update_col = True
        update_lon = True

    # Get ref, action, take action
    if step < 6000:
        actions = np.array([collective_pid(obs), # + agent_col.get_action(obs, ref),
                           trim_action[1] + agent_lon.get_action(obs, ref)])
    else:
        actions = np.array([trim_action[0] + agent_col.get_action(obs, ref),
                            trim_action[1] + agent_lon.get_action(obs, ref)])

    if excitation_phase:
        actions += excitation[step]
    next_obs, _, done = env.step(actions)
    next_ref = env.get_ref()

    # Update RLS estimator,
    RLS.update(obs, actions, next_obs)

    # Cyclic
    if update_lon:
        rewards[1], dr_ds = agents[1].get_reward(next_obs, ref)
        F, G = agents[1].get_transition_matrices(RLS)
        agents[1].update_networks(obs, next_obs, ref, next_ref, dr_ds, F, G)
    else:
        rewards[1] = 0

    # Collective:
    if update_col:
        rewards[0], dr_ds = agents[0].get_reward(next_obs, ref)
        F, G = agents[0].get_transition_matrices(RLS)
        agents[0].update_networks(obs, next_obs, ref, next_ref, dr_ds, F, G)
    else:
        rewards[0] = 0

    # Log data
    stats.append({'t': env.t,
                  'x': obs[0],
                  'z': obs[1],
                  'u': obs[2],
                  'w': obs[3],
                  'theta': obs[4],
                  'q': obs[5],
                  'reference': ref,
                  'collective': actions[0],
                  'cyclic': actions[1],
                  'r1': rewards[0],
                  'r2': rewards[1]})

    # Save NN and RLS weights
    weight_stats['t'].append(env.t)
    if step % 10 == 0:
        for agent in agents:
            for nn, network_name in zip(agent_col.networks, network_sequence):
                for layer, layer_name in zip(nn.parameters(), layer_sequence):
                    weight_stats[agent.control_channel_str]['nn'][network_name][layer_name].append(layer.detach().numpy().ravel())
            tmp_F, tmp_G = agent.get_transition_matrices(RLS)
            weight_stats[agent.control_channel_str]['rls']['F'].append(tmp_F.detach().numpy().ravel())
            weight_stats[agent.control_channel_str]['rls']['G'].append(tmp_G.detach().numpy().ravel())

    # Next step
    obs = next_obs
    ref = next_ref

    if np.isnan(actions).any():
        print("NaN encounted in actions at timestep", step, " -- ", actions)
        break

    if done:
        break

t2 = time.time()
print("Training time: ", t2 - t_start)
stats = pd.DataFrame(stats)
plot_stats_3dof(stats, pitch_rate=False)
print("Post-processing-time: ", time.time() - t2)
