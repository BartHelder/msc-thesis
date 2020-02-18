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
from PID import CollectivePID


class Logger:

    def __init__(self, name, n_agents):
        self.name = name
        self.state_history = []
        self.timestamp = datetime.datetime.now()
        self.agents = defaultdict(dict)
        for n in range(n_agents):
            self.agents['agent_'+str(n)]

    def load(self, filepath):
        return


t0 = time.time()
torch.manual_seed(0)
np.random.seed(0)

# Some parameters
params_cyclic = {}
params_collective = {}

V_INITIAL = 20
TRACKED_STATE = 5
AC_STATES = [4]
NN_INPUTS = len(AC_STATES)+1
NN_HIDDEN = 10
NN_STDEV = 1
state_indices = AC_STATES + [TRACKED_STATE]
reward_weight = 1
lr_actor = 0.4
lr_critic = 0.4
gamma = 0.95
tau = 0.01
dt = 0.01
t_max = 120
n_steps = int(t_max / dt)

# EnvironmentSt
config_path = "/home/bart/PycharmProjects/msc-thesis/config_3dof.json"
env = Helicopter3DOF(dt=dt, t_max=t_max)
env.setup_from_config(task="sinusoid", config_path=config_path)
obs, trim_action = env.reset(v_initial=V_INITIAL)
ref = env.get_ref()
dr_ds = torch.zeros((1, len(AC_STATES)+1))

# incremental RLS estimator
RLS = RecursiveLeastSquares(**{'state_size': 7,
                               'action_size': 2,
                               'gamma': 1,
                               'covariance': 10**8,
                               'constant': False})

# Agents:
#  Neural networks

agent_col = DHPAgent(control_channel='col',
                     discount_factor=0.95,
                     n_hidden_actor=10,
                     nn_stdev_actor=0.1,
                     learning_rate_actor=0.4,
                     action_scaling=5,
                     n_hidden_critic=10,
                     nn_stdev_critic=0.1,
                     learning_rate_critic=0.4,
                     tau_target_critic=0.01,
                     tracked_state=1,
                     ac_states=[3]
                     )

agent_lon = DHPAgent(control_channel='lon',
                     discount_factor=0.95,
                     n_hidden_actor=10,
                     nn_stdev_actor=1,
                     learning_rate_actor=0.4,
                     action_scaling=10,
                     n_hidden_critic=10,
                     nn_stdev_critic=1,
                     learning_rate_critic=0.4,
                     tau_target_critic=0.01,
                     tracked_state=5,
                     ac_states=[4])
#  PID
#collective_controller = CollectivePID(dt=dt)

agents = [agent_col, agent_lon]

# Excitation signal for the RLS estimator
excitation = np.zeros((1000, 2))
# for j in range(400):
#     #excitation[j, 1] = -np.sin(np.pi * j / 50)
#     #excitation[j + 400, 1] = np.sin(2 * np.pi * j / 50) * 2
# excitation = np.deg2rad(excitation)

# Bookkeeping
excitation_phase = True
done = False
step = 0
rewards = np.zeros(2)
stats = []
t_start = time.time()

#  Main loop
for step in range(n_steps):

    # Get ref, action, take action
    actions = np.array([trim_action[0] + agent_col.get_action(obs, ref),
                        trim_action[1] + agent_lon.get_action(obs, ref)])
    if excitation_phase:
        actions += excitation[step]
    next_obs, _, done = env.step(actions)
    next_ref = env.get_ref()

    # Update RLS estimator,
    RLS.update(obs, actions, next_obs)

    for agent, i in zip(agents, count()):
        rewards[i], dr_ds = agent.get_reward(next_obs, ref)
        F, G = agent.get_transition_matrices(RLS)
        agent.update_networks(obs, next_obs, ref, next_ref, dr_ds, F, G)

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

    # Next step
    obs = next_obs
    ref = next_ref

    if np.isnan(actions).any():
        print("NaN encounted in actions at timestep", step, " -- ", actions)
        break

    if step == 800:
        excitation_phase = False

    if done:
        break

t2 = time.time()
print("Training time: ", t2 - t_start)
stats = pd.DataFrame(stats)
plot_stats_3dof(stats)
print("Post-processing-time: ", time.time() - t2)
