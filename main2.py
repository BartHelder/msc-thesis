import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import RecursiveLeastSquares
from heli_models import Helicopter3DOF
from plotting import plot_stats_3dof
from PID import CollectivePID


class DHPCritic(nn.Module):
    def __init__(self, ni=2, nh=8, std=0.1):
        super(DHPCritic, self).__init__()
        self.fc1 = nn.Linear(ni, nh, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0, std=std)
        self.fc2 = nn.Linear(nh, ni, bias=False)
        nn.init.normal_(self.fc2.weight, mean=0, std=std)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class DHPActor(nn.Module):
    def __init__(self, ni=2, nh=8, std=0.1, scaling=10):
        super(DHPActor, self).__init__()
        self.fc1 = nn.Linear(ni, nh, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0, std=std)
        self.fc2 = nn.Linear(nh, 1, bias=False)
        nn.init.normal_(self.fc2.weight, mean=0, std=std)
        self.scale = np.deg2rad(scaling)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.scale * x
        return x

class Logger:

    def __init__(self, n_agents):
        self.state_history = []
        self.agent
        self.off

    def load(self):
        return

def augment_state(obs, ref):
    return torch.tensor([obs[x] for x in AC_STATES] + [ref[TRACKED_STATE] - obs[TRACKED_STATE]], requires_grad=True)


if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)

    # Some parameters
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
    gamma = 0.8
    tau = 0.01
    dt = 0.01
    t_max = 120
    n_steps = int(t_max / dt)

    # Environment
    config_path = "/home/bart/PycharmProjects/msc-thesis/config_3dof.json"
    env = Helicopter3DOF(dt=dt, t_max=t_max)
    env.setup_from_config(task="sinusoid", config_path=config_path)
    obs, trim_action = env.reset(v_initial=V_INITIAL)
    dr_ds = torch.zeros((1, len(AC_STATES)+1))

    # incremental RLS estimator
    RLS = RecursiveLeastSquares(**{'state_size': 7,
                                   'action_size': 2,
                                   'gamma': 1,
                                   'covariance': 10**8,
                                   'constant': False})

    # Agents:
    #  Neural networks
    actor = DHPActor(ni=NN_INPUTS, nh=NN_HIDDEN, std=NN_STDEV)
    critic = DHPCritic(ni=NN_INPUTS, nh=NN_HIDDEN, std=NN_STDEV)
    target_critic = copy.deepcopy(critic)
    #  PID
    collective_controller = CollectivePID(dt=dt)

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
    stats = []

    ########## Main loop:
    for step in range(n_steps):

        # Get ref, action, take action
        ref = env.get_ref()
        aug = augment_state(obs, ref)
        cyclic = actor.forward(aug)
        collective = collective_controller(obs)
        actions = np.array([collective, trim_action[1] + cyclic.data])
        if excitation_phase:
            actions += excitation[step]
        next_obs, _, done = env.step(actions)
        if np.isnan(next_obs).any():
            print("NaN encounted in next_obs at timestep", step)
            break
        # Process transition based on next state and current reference
        tracking_error = ref[TRACKED_STATE] - next_obs[TRACKED_STATE]  # r_{t+1} = f(s_{t+1}, sRef_{t})
        reward = -tracking_error**2 * reward_weight
        dr_ds[:, -1] = 2 * tracking_error * reward_weight

        # Augment next state - input for target critic
        next_ref = env.get_ref()
        next_aug = augment_state(next_obs, next_ref)

        # Update RLS estimator,
        RLS.update(obs, actions, next_obs)
        F = torch.tensor(RLS.gradient_state()[np.ix_(state_indices, state_indices)], dtype=torch.float)
        G = torch.tensor(RLS.gradient_action()[np.ix_(state_indices, [1])],  dtype=torch.float)

        # Forward passes...
        lambda_t1 = critic.forward(aug)
        with torch.no_grad():
            lambda_t2 = target_critic.forward(next_aug)

        # Backpropagate raw action through actor network
        cyclic.backward()
        da_ds = aug.grad

        # From DHP definition:
        target = dr_ds + gamma * lambda_t2
        error_critic = lambda_t1 - target.mm(F + G.mm(da_ds.unsqueeze(0)))

        # Backpropagate error_critic through critic network and update weights
        lambda_t1.backward(error_critic.squeeze())
        with torch.no_grad():
            for wa, wc in zip(actor.parameters(), critic.parameters()):
                wa.data.sub_(wa.grad.data * (-target.mm(G).squeeze(dim=0)) * lr_actor)
                wc.data.sub_(wc.grad.data * lr_critic)
            critic.zero_grad()
            actor.zero_grad()
            target_critic.zero_grad()

        # Update target network
        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

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
                      'r1': 0,
                      'r2': reward})

        obs = next_obs

        if step == 800:
            excitation_phase = False

        if done:
            break

    stats = pd.DataFrame(stats)
    plot_stats_3dof(stats)