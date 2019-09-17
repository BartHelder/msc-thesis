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

    def __init__(self, tau, k_beta, task, name='heli'):
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
        self.q_threshold = np.deg2rad(15)
        self.qe_threshold = np.deg2rad(10)

        self.task = task

        high = np.array([
            self.q_threshold * 2,
            np.deg2rad(1),
            self.qe_threshold * 2])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)


    def step(self, u_cyclic, virtual=False):
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
        q_ref = self.task.step()

        # Get state derivatives
        q_dot = -self.th_iy * (u_cyclic - a1)
        a1_dot = (-1 / self.tau) * (a1 + 16 * q / (self.gamma * self.omega))

        # Integration
        q = q + self.dt * q_dot
        a1 = a1 + self.dt * a1_dot

        if not virtual:
            self.state = [q, a1]

        qe = (q - q_ref)
        reward = (-1/2 * (qe / self.q_threshold)**2)

        done = True if self.task.t >= self.max_episode_length else False

        return np.array([q, a1, qe]), reward, done

    def get_environment_transition_function(self):
        return np.array([-self.th_iy * self.dt,
                         0,
                         -self.th_iy * self.dt])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """"
        State variabloes: q, al, theta
        """

        self.task.reset()

        q_0 = np.random.uniform(low=-1, high=1) * np.deg2rad(0.1)
        a1 = np.random.uniform(low=-1, high=1) * np.deg2rad(0.01)
        self.state = [q_0, a1]

        return np.array([q_0, a1, q_0])


class Heli3DOF:

    def __init__(self, dt=0.02):

        self.dt = dt

        self.g = 9.81
        self.cl_alpha = 5.7  # NACA0012
        self.volh = .075  # blade solidity parameter
        self.lok = 6
        self.cds = 1.5
        self.mass = 2200  #kg
        self.rho = 1.225  #kg/m^3
        self.v_tip = 200  #m/s
        self.rotor_radius = 7.32 #m
        self.iy = 10615  # moment of inertia, kg*m^2
        self.mast = 1
        self.omega = self.v_tip / self.rotor_radius
        self.area = np.pi * self.rotor_radius**2
        self.tau = 0.1

        self.state = np.array([0, 0, 0, 0, 0, 0, 0])
        # self.state = np.array([x, z, u, w, pitch, q, lambda_i])

    def reset(self, v):

        return self.state

    def trim(self, v_trim=3):
        """
        Trim the helicopter at a certain initial velocity, sets the state correctly and returns controls required to
        keep the trimmed velocity.
        :param v_trim:
        :return:
        """

    def step(self, actions, virtual=False):

        theta_0, theta_s = actions                  # collective, cyclic pitch
        x, z, u, w, pitch, q, lambda_i = self.state
        q_diml = q / self.omega                     # dimensionless pitch rate
        v_diml = np.sqrt(u**2 + w**2) / self.v_tip  # dimensionless speed
        alpha_c = theta_s - np.arctan2(w, u)

        mu = v_diml * np.cos(alpha_c)  # tip speed ratio
        lambda_c = v_diml * np.sin(alpha_c)

        # Flapping calculations
        a1 = (-16*q_diml / self.lok + 8/3 * mu * theta_0 - 2 * mu * (lambda_c + lambda_i)) / (1 - 1/2 * mu**2)

        # Now calculate the two different thrust coefficients
        ct_bem = self.cl_alpha * self.volh / 4 * (2/3 * theta_0 * (1 + (3/2) * mu**2) - (lambda_c + lambda_i))
        alpha_d = alpha_c - a1
        ct_glau = 2 * lambda_i * np.sqrt((v_diml * np.cos(alpha_d))**2 + (v_diml * np.sin(alpha_d) + lambda_i)**2)

        # Equations of motion
        thrust = ct_bem * self.rho * self.v_tip**2 * self.area
        helling = theta_s - a1
        vv = v_diml * self.v_tip

        x_dot = u * np.cos(pitch) + w * np.sin(pitch)
        z_dot = -w
        u_dot = -self.g * np.sin(pitch) - self.cds / self.mass * .5 * self.rho * u * vv + thrust / self.mass * np.sin(helling) - q * w
        w_dot = self.g * np.cos(pitch) - self.cds / self.mass * .5 * self.rho * w * vv - thrust / self.mass * np.cos(helling) + q * u
        pitch_dot = q
        q_dot = -thrust * self.mast / self.iy * np.sin(helling)
        lambda_i_dot = (ct_bem - ct_glau) / self.tau

        # Numerical integration
        u += u_dot * self.dt
        w += w_dot * self.dt
        q+= q_dot * self.dt
        pitch+= pitch_dot * self.dt
        x+= x_dot * self.dt
        z+= z_dot * self.dt
        lambda_i+= lambda_i_dot * self.dt

        state = np.array([x, z, u, w, pitch, q, lambda_i])
        # Save results:
        if not virtual:
            self.state = state

        # Todo: implement reward and done
        reward = 0
        done = False

        return state, reward, done


if __name__ == "__main__":
    env = Heli3DOF(dt=0.02)




# for i=1:aantal
#
# # Collective:
# hwens = 25
# c = u * np.sin(pitch) - w * np.cos(pitch)
# h = -z
# cwens = .1 * (hwens - h)
# collectgrd = 5 + 2 * (cwens - c) + 0.2 * corrc
# collect = collectgrd * pi / 180
#
# # Cyclic pitch:
# if t < 90
#     % law
#     1
#     for helic.pitch
#         uwens = 50
#     pitchwens = -.005 * (uwens - u) - .0005 * corr % in rad
#     xeind = x
#     pitcheind = pitchwens
# else
#     % law
#     2
#     for helic.pitch
#         xxeind = xeind(900)
#     pitcheeind = pitcheind(900)
#     pitchwens = -.001 * (xxeind + 2000 - x) + .02 * u % in rad
#     if pitchwens < pitcheeind
#         pitchwens = pitcheeind % in rad
#     end
# end
# longitgrd = (.2 * (pitch - pitchwens) + .8 * q) * 180 / pi % in deg
# if longitgrd > 10
#     longitgrd = 10
# end
# if longitgrd < -10
#     longitgrd = -10
# end
# longit = longitgrd * pi / 180 % in rad



