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
from gym import spaces
from gym.utils import seeding



class SimpleHelicopter:

    def __init__(self, tau, k_beta, name='heli'):
        self.dt = 0.02
        self.max_episode_length = 120
        self.episode_ticks = self.max_episode_length / self.dt

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
        self.q_threshold = np.deg2rad(5)
        self.qe_threshold = np.deg2rad(3)

        high = np.array([
            self.q_threshold * 2,
            self.qe_threshold * 2])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def step(self, u_cyclic, q_ref, virtual=False):
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

        q, a1 = self.state

        # Get state derivatives
        q_dot = -self.th_iy * (u_cyclic - a1)
        a1_dot = (-1 / self.tau) * (a1 + 16 * q / (self.gamma * self.omega))

        # Integration
        q = q + self.dt * q_dot
        a1 = a1 + self.dt * a1_dot

        if not virtual:
            self.state = [q, a1]

        qe = (q - q_ref)
        done = False
        reward = -1 / 2 * qe ** 2
        if qe > self.qe_threshold:
            reward -= 10

        return np.array([q, a1, qe]), reward, done

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """"
        State variabloes: q, al, theta
        """

        q_0 = np.random.uniform(low=-1, high=1) * np.deg2rad(1)
        a1 = 0
        self.t = 0
        self.state = [q_0, a1]

        return np.array([q_0, a1, 0])

    def render(self, t):
        plt.scatter(t, self.state[0])
        plt.pause(0.04)
        plt.show()
