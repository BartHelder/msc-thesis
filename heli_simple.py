import gym
import numpy as np
import itertools
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class SimpleHelicopter(gym.Env):

    def __init__(self, tau, k_beta):
        self.dt = 0.001
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

        self.state = None

    def step(self, u_cyclic):
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

        state = self.state
        q, al = state
        al_dot = (-1 / self.tau) * (al + 16 * q / (self.gamma * self.omega))
        q_dot = -self.th_iy * (u_cyclic - al)

        al = al + self.dt * al_dot
        q = q + self.dt * q_dot

        self.state = np.array([q, al])
        reward = 1

        return self.state, reward

    def reset(self):
        """"
        State variabloes: q, al
        """
        self.state = (0, 0)
        return self.state

def controller(env):

    state = env.state
    K = np.array([-10, -0.44])
    action = np.matmul(K, state)
    return action

TeeteredHeli = SimpleHelicopter(tau=0.05, k_beta=0)
RigidHeli = SimpleHelicopter(tau=0.25, k_beta=460000)
envs = [TeeteredHeli, RigidHeli]

for env in envs:
    stats = []
    for i_episode in range(1):
        state = env.reset()
        action = 0
        t_end = 17
        tick = env.dt

        for t in np.arange(0, t_end, tick):

            next_state, reward = env.step(action)
            q_ref = -1 * np.pi / 180 * np.sin(t * 2 * np.pi / 5)
            k = (-16 / (env.gamma * env.omega))  # Steady-state value of q for unit cyclic
            next_action = q_ref * k
            # next_action = (q_ref - controller(env))

            stats.append({'t': t, 'q_ref': q_ref, 'u': action, 'y': state[0]})
            if t > 20:
                break
            state, action = next_state, next_action

        df = pd.DataFrame(stats)
        sns.set()
        ax = plt.figure(figsize=(10, 6))
        plt.plot(df['t'], df['q_ref'] * 180 / np.pi, label='q_ref')
        plt.plot(df['t'], df['y'] * 180 / np.pi, label='q')

        plt.legend()
        plt.show()
