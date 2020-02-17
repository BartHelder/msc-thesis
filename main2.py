import copy
import datetime
import time
import logging
from collections import defaultdict

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

    def __init__(self, name, n_agents):
        self.name = name
        self.state_history = []
        self.timestamp = datetime.datetime.now()
        self.agents = defaultdict(dict)
        for n in range(n_agents):
            self.agents['agent_'+str(n)]

    def load(self, filepath):
        return


class Agent:

    def __init__(self, control_channel: str, discount_factor,
                 n_hidden_actor: int, nn_stdev_actor, learning_rate_actor, action_scaling,
                 n_hidden_critic: int, nn_stdev_critic, learning_rate_critic, tau_target_critic,
                 tracked_state: int, ac_states: list):
        """
        Describe this
        :param control_channel:
        :param n_hidden_actor:
        :param nn_stdev_actor:
        :param discount_factor_actor:
        :param action_scaling:
        :param n_hidden_critic:
        :param nn_stdev_critic:
        :param discount_factor_critic:
        :param target_critic_tau:
        :param tracked_state:
        :param ac_states:
        """

        self.channel = control_channel
        self.tracked_state = tracked_state
        self.ac_states = ac_states
        self.n_inputs = len(ac_states)+1
        self.target_critic_tau = tau_target_critic

        self.actor = DHPActor(ni=self.n_inputs, nh=n_hidden_actor, std=nn_stdev_actor, scaling=action_scaling)
        self.critic = DHPCritic(ni=self.n_inputs, nh=n_hidden_critic, std=nn_stdev_critic)
        self.target_critic = copy.deepcopy(self.critic)

        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma = discount_factor

    def get_action(self, state, ref):
        augmented_state = self.augment_state(state, ref)
        with torch.no_grad():
            action = self.actor.forward(augmented_state)
        return action.data

    def augment_state(self, state, reference):
        augmented_state = [state[x] for x in self.ac_states] + [reference[self.tracked_state] - obs[self.tracked_state]]
        return torch.tensor(augmented_state, requires_grad=True)

    def update_networks(self, state, next_state, ref, next_ref, dr_ds, F, G):
        """
        Update the actor, critic and target critic model by doing forward and backward passes through the respective
        neural networks according to the Dual Heuristic Dynamic Programming (DHP) strategy.
        :param state:
        :param next_state:
        :param dr_ds:
        :param F:
        :param G:
        :return:
        """
        augmented_state = self.augment_state(state, ref)
        next_augmented_state = self.augment_state(next_state, next_ref)

        # Forward passes...
        action = self.actor.forward(augmented_state)
        lambda_t1 = self.critic.forward(augmented_state)
        lambda_t2 = self.target_critic.forward(next_augmented_state)

        # Backpropagate raw action through actor network
        action.backward()
        da_ds = augmented_state.grad

        # From DHP definition:
        target = dr_ds + gamma * lambda_t2
        error_critic = lambda_t1 - target.mm(F + G.mm(da_ds.unsqueeze(0)))

        # Backpropagate error_critic through critic network and update weights
        lambda_t1.backward(error_critic.squeeze())
        # Make sure these calculations don't affect the actual gradients by wrapping them in no_grad()
        with torch.no_grad():
            for wa, wc in zip(self.actor.parameters(), self.critic.parameters()):
                # .sub_() is in-place substraction - fast en memory-efficient
                wa.data.sub_(wa.grad.data * (-target.mm(G).squeeze(dim=0)) * self.learning_rate_actor)
                wc.data.sub_(wc.grad.data * self.learning_rate_critic)
            # In PyTorch, gradients accumulate rather than overwrite, so after updating they must be zeroed:
            self.critic.zero_grad()
            self.actor.zero_grad()
            self.target_critic.zero_grad()  # I don't think these have a value inside of them but just to be sure...

        # Update target network - copy_() is a fast and memory-unintensive value overwrite
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


if __name__ == "__main__":

    t0 = time.time()
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
    dr_ds = torch.zeros((1, len(AC_STATES)+1))

    # incremental RLS estimator
    RLS = RecursiveLeastSquares(**{'state_size': 7,
                                   'action_size': 2,
                                   'gamma': 1,
                                   'covariance': 10**8,
                                   'constant': False})

    # Agents:
    #  Neural networks
    agent = Agent(control_channel='cyclic',
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
    t_start = time.time()
    print("Setup time: ", t_start - t0)
    ########## Main loop:
    for step in range(n_steps):

        # Get ref, action, take action
        ref = env.get_ref()
        cyclic = agent.get_action(obs, ref)
        collective = collective_controller(obs)
        actions = np.array([collective, trim_action[1] + cyclic])
        if excitation_phase:
            actions += excitation[step]
        next_obs, _, done = env.step(actions)

        # Process transition based on next state and current reference
        tracking_error = ref[TRACKED_STATE] - next_obs[TRACKED_STATE]  # r_{t+1} = f(s_{t+1}, sRef_{t})
        reward = -tracking_error**2 * reward_weight
        dr_ds[:, -1] = 2 * tracking_error * reward_weight

        # Update RLS estimator,
        RLS.update(obs, actions, next_obs)
        F = torch.tensor(RLS.gradient_state()[np.ix_(state_indices, state_indices)], dtype=torch.float)
        G = torch.tensor(RLS.gradient_action()[np.ix_(state_indices, [1])],  dtype=torch.float)

        # Get new ref from environment, update actor and critic networks
        next_ref = env.get_ref()
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
                      'r1': 0,
                      'r2': reward})

        obs = next_obs

        if np.isnan(actions).any():
            print("NaN encounted in actions at timestep", step)
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
