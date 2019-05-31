import gym
import numpy as np
import itertools
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import plotting
from gym.envs.classic_control import cartpole

class SimpleHelicopter():

    def __init__(self, tau, k_beta, name):
        self.dt = 0.01
        self.mass = 2200
        self.h = 1
        self.tau = tau
        self.k_beta = k_beta
        self.iy = 10625
        self.gamma = 6
        self.v_tip = 200
        self.r_blade = 7.32
        self.omega = (self.v_tip / self.r_blade)
        self.th_iy = (self.mass * 9.81 * self.h + 3 / 2 * self.k_beta) / self.iy
        self.name = name
        self.state = None
        self.theta = 0
        self.theta_threshold = 30 * np.pi / 180
        self.q_space = (np.arange(-5, 5.025, 0.025)) * np.pi / 180
        self.qe_space = (np.arange(-2, 2.01, 0.01)) * np.pi / 180
        self.a1_space = (np.arange(-1, 1.05, 0.05)) * np.pi / 180
        self.action_space = np.arange(-5, 5.01, 0.5) * np.pi / 180

    def step(self, action, q_ref):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        u_cyclic = self.action_space[action]
        q, a1 = self.state
        a1_dot = (-1 / self.tau) * (a1 + 16 * q / (self.gamma * self.omega))
        q_dot = -self.th_iy * (u_cyclic - a1)

        a1 = a1 + self.dt * a1_dot
        q = q + self.dt * q_dot

        qd = min(np.searchsorted(self.q_space, q), 400)
        qed = min(np.searchsorted(self.qe_space, q-q_ref), 400)
        a1d = np.searchsorted(self.a1_space, a1)

        q_discretized = self.q_space[qd]
        qe_discretized = self.qe_space[qed]
        a1_discretized = self.a1_space[a1d]

        self.theta += self.dt * q_discretized
        self.state = np.array([q_discretized, a1_discretized])

        done = False
        if self.theta < -self.theta_threshold or self.theta > self.theta_threshold \
               or qe_discretized < min(self.qe_space) or qe_discretized > max(self.qe_space):
            reward = -10
            done = True
        else:
            reward = -qe_discretized**2 / 2

        return (qd, qed, a1d), reward, done, {}

    def reset(self):
        """"
        State variabloes: q, al
        """
        self.state = np.array([0, 0])
        self.theta = 0

        return 0, 0, 0




