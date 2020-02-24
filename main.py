import itertools

import time
import numpy as np
import pandas as pd
import torch

from agents import DHPAgent
from model import RecursiveLeastSquares
from heli_models import Helicopter3DOF
from heli_models import Helicopter6DOF
from plotting import plot_neural_network_weights_2, plot_stats_6dof, plot_policy_function, plot_rls_stats
from PID import LatPedPID, CollectivePID6DOF

# + np.sin(2 * np.pi * t / 10)

def get_ref(obs, t, z_ref_start):
    ref = np.nan * np.ones_like(observation)
    if t < 60:
        ref[11] = 0
        qref = np.deg2rad(10 * (np.sin(np.pi * t / 10)))
        if np.rad2deg(obs[7]) > 20:
            qref -= 2 * abs(obs[7]-np.deg2rad(20))
        elif np.rad2deg(obs[7]) < -20:
            qref += 2 * abs(obs[7] + np.deg2rad(20))
    else:
        qref = np.clip(-obs[7] * 0.5, -np.deg2rad(5), np.deg2rad(5))
        ref[11] = (z_ref_start - 20 * np.sin((t - 60) / 10))
    ref[4] = qref
    zref = 0
    #ref[11] = zref
    max_zref_deviation = 5

    return ref


def envelope_limits_reached(obs, limit_pitch=89, limit_roll=89):
    limit_reached = False
    which = None
    if abs(np.rad2deg(obs[7])) > limit_pitch:
        limit_reached = True
        which = 'pitch angle > ' + str(limit_pitch) + ' deg'
    elif abs(np.rad2deg(obs[6])) > limit_roll:
        limit_reached = True
        which = 'roll angle > ' + str(limit_roll) + ' deg'

    return limit_reached, which


t0 = time.time()
torch.manual_seed(167)
np.random.seed(0)

# Some parameters
agent_parameters = {'col':
                    {'control_channel': 'col',
                     'discount_factor': 0.9,
                     'n_hidden_actor': 10,
                     'nn_stdev_actor': 0.1,
                     'learning_rate_actor': 0.1,
                     'action_scaling': None,
                     'n_hidden_critic': 10,
                     'nn_stdev_critic': 0.1,
                     'learning_rate_critic': 0.1,
                     'tau_target_critic': 0.01,
                     'tracked_state': 11,
                     'ac_states': [2],
                     'reward_weight': 0.1},
                    'lon':
                    {'control_channel': 'lon',
                     'discount_factor': 0.9,
                     'n_hidden_actor': 10,
                     'nn_stdev_actor': 0.75,
                     'learning_rate_actor': 0.4,
                     'action_scaling': None,
                     'n_hidden_critic': 10,
                     'nn_stdev_critic': 0.75,
                     'learning_rate_critic': 0.4,
                     'tau_target_critic': 0.01,
                     'tracked_state': 4,
                     'ac_states': [7]}}
rls_parameters = {'state_size': 15,
                  'action_size': 2,
                  'gamma': 1,
                  'covariance': 10**8,
                  'constant': False}
V_INITIAL = 20
dt = 1 / 100
t_max = 120
n_steps = int(t_max / dt)

# EnvironmentSt
config_path = "/home/bart/PycharmProjects/msc-thesis/config_3dof.json"
env = Helicopter6DOF(dt=dt, t_max=t_max)
trim_state, trim_actions = env.trim(trim_speed=V_INITIAL, flight_path_angle=0, altitude=0)

# incremental RLS estimator
RLS = RecursiveLeastSquares(**rls_parameters)

# Agents:
agent_col = DHPAgent(**agent_parameters['col'])
agent_lon = DHPAgent(**agent_parameters['lon'])
agents = [agent_col, agent_lon]
# Create controllers
LatPedController = LatPedPID(config_path='config_6dof.json',
                             phi_trim=trim_state[6],
                             lat_trim=trim_actions[2],
                             pedal_trim=trim_actions[3])
ColController = CollectivePID6DOF(col_trim=trim_actions[0],
                                  h_ref=0,
                                  dt=dt,
                                  proportional_gain=0.0025
                                  )

# Excitation signal for the RLS estimator
excitation = np.zeros((1000, 2))
# for j in range(400):
#     #excitation[j, 1] = -np.sin(np.pi * j / 50)
#     #excitation[j + 400, 1] = np.sin(2 * np.pi * j / 50) * 2
excitation = np.deg2rad(excitation)

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


# Bookkeeping
excitation_phase = False
done = False
rewards = np.zeros(2)
t_start = time.time()
update_col = False
update_lon = True
observation = trim_state.copy()
ref = get_ref(observation, env.t, 0)
step = 0
save_weights = True
z_ref_start = 0
while not done:

    if step == 6000:
        z_ref_start = stats[-1]['z']
        update_col = True
        update_lon = True

    # Get ref, action, take action
    lateral_cyclic, pedal = LatPedController(observation)
    if step < 6000:
        actions = np.array([ColController(observation),
                           trim_actions[1]-0.5 + agent_lon.get_action(observation, ref),
                           lateral_cyclic,
                           pedal])
    else:
        actions = np.array([trim_actions[0]-0.5 + agent_col.get_action(observation, ref),
                            trim_actions[1]-0.5 + agent_lon.get_action(observation, ref),
                            lateral_cyclic,
                            pedal])

    # Add excitations (RLS windup phase) and trim values
    if excitation_phase:
        actions += excitation[step]

    actions = np.clip(actions, 0, 1)

    # Take step in the environment
    next_observation, _, done = env.step(actions)
    next_ref = get_ref(next_observation, env.t, z_ref_start)
    # Update RLS estimator,
    RLS.update(observation, actions[:2], next_observation)

    # Cyclic
    if update_lon:
        rewards[1], dr_ds = agents[1].get_reward(next_observation, ref)
        F, G = agents[1].get_transition_matrices(RLS)
        agents[1].update_networks(observation, next_observation, ref, next_ref, dr_ds, F, G)
    else:
        rewards[1] = 0

    # Collective:
    if update_col:
        rewards[0], dr_ds = agents[0].get_reward(next_observation, ref)
        F, G = agents[0].get_transition_matrices(RLS)
        agents[0].update_networks(observation, next_observation, ref, next_ref, dr_ds, F, G)
    else:
        rewards[0] = 0

    # Log data
    stats.append({'t': env.t,
                  'u': observation[0],
                  'v': observation[1],
                  'w': observation[2],
                  'p': observation[3],
                  'q': observation[4],
                  'r': observation[5],
                  'phi': observation[6],
                  'theta': observation[7],
                  'psi': observation[8],
                  'x': observation[9],
                  'y': observation[10],
                  'z': observation[11],
                  'ref': ref.copy(),
                  'col': actions[0],
                  'lon': actions[1],
                  'lat': actions[2],
                  'ped': actions[3],
                  'r1': rewards[0],
                  'r2': rewards[1]})

    # Save NN and RLS weights
    if save_weights and step % 10 == 0:
        weight_stats['t'].append(env.t)
        for agent in agents:
            for nn, network_name in zip(agent_col.networks, network_sequence):
                for layer, layer_name in zip(nn.parameters(), layer_sequence):
                    weight_stats[agent.control_channel_str]['nn'][network_name][layer_name].append(layer.detach().numpy().ravel().copy())
            tmp_F, tmp_G = agent.get_transition_matrices(RLS)
            weight_stats[agent.control_channel_str]['rls']['F'].append(tmp_F.detach().numpy().ravel().copy())
            weight_stats[agent.control_channel_str]['rls']['G'].append(tmp_G.detach().numpy().ravel().copy())

    if envelope_limits_reached(observation)[0]:
        print("Save envelope limits reached, stopping simulation. ")
        print("Cause of violation: " + envelope_limits_reached(observation)[1])
        done = True

    # Next step..
    observation = next_observation
    ref = next_ref
    step += 1

    if np.isnan(actions).any():
        print("NaN encounted in actions at timestep", step, " -- ", actions)
        done = True

t2 = time.time()
print("Training time: ", t2 - t_start)
stats = pd.DataFrame(stats)
plot_stats_6dof(stats, results_only=False)
