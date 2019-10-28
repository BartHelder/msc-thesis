import gym
import numpy as np
import itertools
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from gym import spaces
from typing import Union
from gym.utils import seeding
from tasks import SimpleTrackingTask, HoverTask


class Helicopter1DOF:

    def __init__(self, tau, k_beta, task, name='heli'):
        self.dt = 0.02
        self.max_episode_length = 120
        self.episode_ticks = self.max_episode_length / self.dt
        self.n_actions = 1

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

        self.stats = {"tau": tau,
                      "k_beta": k_beta}

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
        reward = (-1 / 2 * (qe / self.q_threshold) ** 2)

        done = True if self.task.t >= self.max_episode_length else False

        return np.array([q, a1, qe]), reward, done


    def get_environment_transition_function(self):

        """ Returns the environment transition derivative ds/da
        In this case, exactly
        Obtainted by differentiating the system wrt a

        """

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


class Helicopter3DOF:

    def __init__(self, task, dt=0.01, t_max=120):
        self.task = task
        self.dt = dt
        self.max_episode_length = t_max
        self.episode_ticks = self.max_episode_length / self.dt
        self.n_actions = 2

        self.g = 9.81
        self.cl_alpha = 5.7  # NACA0012 airfoil
        self.blade_solidity = .075  # blade solidity parameter
        self.lock_number = 6
        self.cds = 1.5  # equivalent flat plate area
        self.mass = 2200  # kg
        self.rho = 1.225  # kg/m^3
        self.v_tip = 200  # m/s
        self.rotor_radius = 7.32  # m
        self.iy = 10615  # moment of inertia, kg*m^2
        self.mast = 1  # mast height, m
        self.omega = self.v_tip / self.rotor_radius
        self.area = np.pi * self.rotor_radius ** 2
        self.tau = 0.1
        # self.state = np.array([x, z, u, w, pitch_fuselage, q, lambda_i])
        self.state = np.array([0, 0, 0, 0, 0, 0, 0])
        self.t = 0

        self.stats = {}

    def reset(self, v_initial=0.5):

        trimmed_controls, trimmed_state = self._trim(v_initial)
        self.state = trimmed_state
        self.t = 0

        return self.state

    def step(self, actions, virtual=False, **kwargs):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).
        :param actions: Set of control inputs: collective and cyclic pitch
        :param virtual: If true, calculates the results of the action but does not save the resulting state
        :param kwargs:
        :return:
        """

        ref = self.task.step()

        collective, cyclic_pitch = actions
        x, z, u, w, pitch, q, lambda_i = self.state
        q_diml = q / self.omega  # dimensionless pitch rate
        v_diml = np.sqrt(u**2 + w**2) / self.v_tip  # dimensionless speed
        alpha_c = cyclic_pitch - np.arctan2(w, u)

        mu = v_diml * np.cos(alpha_c)  # tip speed ratio
        lambda_c = v_diml * np.sin(alpha_c)

        # Flapping calculations
        a1 = (-16*q_diml/self.lock_number + (8/3)*mu*collective - 2*mu*(lambda_c+lambda_i)) / (1 - mu**2/2)

        # Now calculate the two different thrust coefficients: blade element method (BEM) and Glauert (glau)
        ct_bem = (self.cl_alpha*self.blade_solidity/4) * ((2/3)*collective*(1+(3/2)*mu**2) - (lambda_c+lambda_i))
        ct_glau = 2*lambda_i*np.sqrt((v_diml*np.cos(alpha_c-a1))**2 + (v_diml*np.sin(alpha_c-a1)+lambda_i)**2)

        # Equations of motion
        thrust = ct_bem * self.rho * self.v_tip**2 * self.area
        vv = v_diml * self.v_tip
        x_dot = u*np.cos(pitch) + w*np.sin(pitch)
        z_dot = -(u*np.sin(pitch) - w*np.cos(pitch))
        u_dot = (-self.g * np.sin(pitch)
                 - self.cds * .5 * self.rho * vv / self.mass * u
                 + thrust / self.mass * np.sin(cyclic_pitch - a1)
                 - q * w)
        w_dot = (self.g * np.cos(pitch)
                 - self.cds * .5 * self.rho * vv / self.mass * w
                 - thrust / self.mass * np.cos(cyclic_pitch - a1)
                 + q * u)
        pitch_dot = q
        q_dot = -thrust * self.mast / self.iy * np.sin(cyclic_pitch - a1)
        lambda_i_dot = (ct_bem - ct_glau) / self.tau

        # Numerical integration
        x += x_dot * self.dt
        z += z_dot * self.dt
        u += u_dot * self.dt
        w += w_dot * self.dt
        pitch += pitch_dot * self.dt
        pitch = np.arctan2(np.sin(pitch), np.cos(pitch))
        q += q_dot * self.dt
        lambda_i += lambda_i_dot * self.dt

        self.t += self.dt

        state = np.array([x, z, u, w, pitch, q, lambda_i])

        if True:
            reward = self._get_reward(goal_state=ref, actual_state=state)
        else:
            reward = 0
        # Save results:
        if not virtual:
            self.state = state

        # If the pitch angle gets too extreme, end the simulation
        done = False
        if np.abs(np.rad2deg(pitch)) > 90:
            done = True

        return state, reward, done

    def get_environment_transition_function(self, h=0.001):
        """
        Returns the instantaneous environment transition derivative ds/da
        :return: numpy array of ds/da of shape (len(s), len(a))
        """
        ds_da1 = (self.step(actions=np.array([1+h, 1]), virtual=True)[0]
                  - self.step(actions=np.array([1, 1]), virtual=True)[0]) / h
        ds_da2 = (self.step(actions=np.array([1, 1+h]), virtual=True)[0]
                  - self.step(actions=np.array([1, 1]), virtual=True)[0]) / h

        return np.array([ds_da1, ds_da2]).T

    def _trim(self, v_trim: Union[float, int] = 3):
        """
        Trim the helicopter at a certain initial velocity, sets the state correctly and returns controls required to
        keep the trimmed velocity.
        :param v_trim: Trim velocity
        :return: Numpy array of trim controls: [collective, cyclic]
        """

        # Forces
        weight = self.mass * self.g
        drag = 1 / 2 * self.rho * v_trim**2 * self.cds
        thrust = np.sqrt(drag ** 2 + weight**2)
        c_t = thrust / (self.rho * self.v_tip**2 * self.area)

        # Body components
        theta_f = np.arctan2(-drag, weight)
        u = v_trim * np.cos(theta_f)
        w = v_trim * np.sin(theta_f)

        # Solving for non-dimensional induced velocity (lambda_i)
        # by equating thrust coefficient above to that found via Glauert's method
        mu = v_trim / self.v_tip  # assuming small angles
        lp = [4, 8 * mu * np.sin(drag / weight), 4 * mu ** 2, 0,
              -c_t ** 2]  # polynomial coefficients, highest order first
        r = np.roots(lp)  # four roots, only one of which is real & positive
        lambda_i = np.real(r[(np.real(r) > 0) & (np.imag(r) == 0)][0])

        # Solve matrix equations to get trim settings
        coef_matrix = np.array([[1 + (3 / 2) * mu ** 2, -8 / 3 * mu], [-mu, 2 / 3 + mu ** 2]])
        b_mat = np.array([[-2 * mu ** 2 * drag / weight - 2 * mu * lambda_i],
                          [4 / self.blade_solidity * c_t / self.cl_alpha + mu * drag / weight + lambda_i]])
        cyclic, collective = np.linalg.solve(coef_matrix, b_mat)

        trimmed_state = np.array([0, 0, u, w, theta_f, 0, lambda_i])

        return np.array([collective[0], cyclic[0]]), trimmed_state

    def _get_reward(self, goal_state, actual_state) -> float:

        P = self.task.selected_states
        Q = self.task.state_weights

        error = (P @ actual_state - goal_state)

        reward = -(error.T @ Q @ error).squeeze()

        return reward


if __name__ == "__main__":
    task = HoverTask()
    env = Helicopter3DOF(dt=0.02, task=task)
    env.reset(v_initial=3)

    sns.set()
    trim_speeds = np.arange(0, 101, 0.1)
    trim_settings = list(map(lambda v: np.rad2deg(env._trim(v_trim=v)[0]), trim_speeds))
    plt.plot(trim_speeds, trim_settings)
    plt.xlabel('Trim speed [m/s]')
    plt.ylabel('Control setting [deg]')
    plt.legend(['collective', 'cyclic'])
    plt.show()

    # collective, cyclic = env.trim(v_trim=10)
    # t = 0
    # t_max = 40
    # step = 1
    # state_history = [env.state]
    # while step < (t_max / env.dt):
    #
    #     if 0.5 <= t <= 1.0:
    #         cyclic = np.deg2rad(1.0)
    #     else:
    #         cyclic = 0
    #     actions = [np.deg2rad(6), cyclic]
    #     state, _, _ = env.step(actions)
    #     state_history.append(state)
    #
    #     step += 1
    #     t += env.dt
    #
    #
    # sh = pd.DataFrame(state_history)
    #
    # t_axis = np.arange(0, t_max, env.dt)
    # x_history = [x[0] for x in state_history]
    #
    # plt.plot(t_axis, x_history)
    # plt.show()

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
