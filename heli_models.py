import gym
import numpy as np
import itertools
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import json
from gym import spaces
from typing import Union
from gym.utils import seeding
from pandas import DataFrame
from scipy.io import loadmat


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

    def __init__(self, dt=0.01, t_max=120):
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
        self.trimmed_state = np.array([0, 0, 0, 0, 0, 0, 0])
        self.task = 'sinusoid'
        self.corr_u = 0
        self.pid_weights = tuple()
        self.ref = None
        self.stats = {}

        self.t = 0.0

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

        ref = self.get_ref()

        collective, cyclic_pitch = np.clip(actions, np.deg2rad([0, -15]), np.deg2rad([10, 15]))
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
        #pitch = np.arctan2(np.sin(pitch), np.cos(pitch))  # Get value clipped between +-180 deg
        q += q_dot * self.dt
        lambda_i += lambda_i_dot * self.dt

        state = np.array([x, z, u, w, pitch, q, lambda_i])
        reward = 0

        # Save results:
        if not virtual:
            #reward = self._get_reward(goal_state=ref, actual_state=state)
            self.t += self.dt
            self.state = state

        # If the pitch angle gets too extreme, end the simulation
        done = False
        if np.abs(np.rad2deg(pitch)) > 90:
            done = True

        return state, reward, done

    def get_ref(self):
        t = self.t + self.dt
        Kp, Ki, Kd = self.pid_weights
        ref = 10
        x = np.pi * t / 10
        # if self.t < 20:
        #     ref = 0
        # elif 20 <= self.t < 70:
        #     ref = 10
        #     x = np.pi * t / 10
        # else:
        #     ref = 15
        #     x = np.pi * t / 10
        h_ref = 0
        if self.task is None:
            return 0

        elif self.task == 'sinusoid':
            qref = np.deg2rad(ref/1.76 * (np.sin(x) + np.sin(2*x)))
            #ref = np.deg2rad(np.sin(2*x) * ref)
            state_ref = np.array([np.nan, -h_ref, np.nan, np.nan, np.nan, qref, np.nan])

        elif self.task == 'velocity':
            u_err = self.ref - self.state[2]
            pitch_ref = np.deg2rad(Kp * u_err + Ki * self.corr_u)
            self.corr_u += u_err * self.dt
            state_ref = np.array([np.nan, h_ref, self.ref, np.nan, pitch_ref, np.nan, np.nan])

        elif self.task == 'stop_over_point':
            x_err = self.ref - self.state[0]
            pitch_ref = np.deg2rad(Kp * x_err + Ki * 0 + Kd * self.state[2])
            # do not accelerate further than trimmed setting if the target is very far away
            pitch_ref = max(pitch_ref, self.trimmed_state[4])
            state_ref = np.array([self.ref, h_ref, np.nan, np.nan, pitch_ref, np.nan, np.nan])

        else:
            raise NotImplementedError("Task type unknown!")

        return state_ref

    def get_environment_transition_function(self, h=0.001):
        """
        Returns the instantaneous environment transition derivative ds/da
        :return: numpy array of ds/da of shape (len(s), len(a))
        """
        ds_da1 = (self.step(actions=np.array([0.1+h, 0.1]), virtual=True)[0]
                  - self.step(actions=np.array([0.1, 0.1]), virtual=True)[0]) / h
        ds_da2 = (self.step(actions=np.array([0.1, 0.1+h]), virtual=True)[0]
                  - self.step(actions=np.array([0.1, 0.1]), virtual=True)[0]) / h

        x1 = np.append(ds_da1, -ds_da1[5])
        x2 = np.append(ds_da2, -ds_da2[5])

        return np.array([[x1, x2]]).T

    def setup_from_config(self, task, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.task = task
        self.dt = config["dt"]
        self._set_pid_weights(config)
        return self.reset(v_initial=config["training"]["trim_speed"])

    def _set_pid_weights(self, config):
        params = config["env"]["tasks"][self.task]
        self.pid_weights = (params["kp"], params["ki"], params["kd"])
        self.ref = params["ref"]

    def reset(self, v_initial=0.5):

        trimmed_controls, trimmed_state = self._trim(v_initial)
        self.state = trimmed_state
        self.t = 0

        return self.state, trimmed_controls

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
        lp = [4, 8 * mu * np.sin(drag / weight), 4 * mu **2, 0,
              -c_t ** 2]  # polynomial coefficients, highest order first
        r = np.roots(lp)  # four roots, only one of which is real & positive
        lambda_i = np.real(r[(np.real(r) > 0) & (np.imag(r) == 0)][0])

        # Solve matrix equations to get trim settings
        coef_matrix = np.array([[1 + (3 / 2) * mu ** 2, -8 / 3 * mu], [-mu, 2 / 3 + mu ** 2]])
        b_mat = np.array([[-2 * mu ** 2 * drag / weight - 2 * mu * lambda_i],
                          [4 / self.blade_solidity * c_t / self.cl_alpha + mu * drag / weight + lambda_i]])
        cyclic, collective = np.linalg.solve(coef_matrix, b_mat)

        trimmed_state = np.array([0, 0, u, w, theta_f, 0, lambda_i])
        self.trimmed_state = trimmed_state
        return np.array([collective[0], cyclic[0]]), trimmed_state

    def _get_reward(self, goal_state, actual_state, clip_reward=True, clip_value=-5.0):

        if self.task is None:
            return 0

        P = np.eye(len(goal_state))[~np.isnan(goal_state)]
        Q = P @ np.diag([0, 0.05, 0, 0, 100, 0, 0]) @ P.T

        error = np.matmul(P, (actual_state - np.nan_to_num(goal_state)))

        reward = -(error.T @ Q @ error).squeeze()
        if clip_reward:
            reward = np.clip(reward, clip_value, 0.0)

        return reward


class Helicopter6DOF:

    def __init__(self, dt, t_max):

        self.dt = dt
        self.t = 0
        self.t_max = t_max
        self.g = 9.80665  # Gravitational acceleration               [m/s^2]
        self.R = 287.05  # Specific gas constant of air             [J/kg/K]
        self.T0 = 288.15  # Sea level temperature in ISA             [K]
        self.h_strat = 11000  # Altitude at which stratosphere begins    [m]
        self.rho0 = 1.2250  # Sea level density in ISA                 [kg/m^3]
        self.standard_atmosphere_gradient = -0.0065  # Standard atmosphere temperature gradient [K/m]

        # General helicopter parameters
        self.mass = 2200  # Helicopter mass   [kg]
        self.W = self.mass * self.g  # Helicopter weight [N]
        self.Ixx = 1433  # Helicopter moment of inertia about x-axis [kg*m^2]
        self.Iyy = 4973  # Helicopter moment of inertia about y-axis [kg*m^2]
        self.Izz = 4099  # Helicopter moment of inertia about z-axis [kg*m^2]

        self.Jxz = 660  # Heli product of inertia about x & z-axis  [kg*m^2]
        self.dxcg = 0  # Displacement of cg along x-axis wrt ref point [m]
        self.dycg = 0  # Displacement of cg along y-axis wrt ref point [m]
        self.dzcg = 0  # Displacement of cg along z-axis wrt ref point [m]

        self.xh = 0.08 - self.dxcg  # Hub x-position relative to cg [m]
        self.yh = 0 - self.dycg  # Hub y-position relative to cg [m]
        self.zh = 1.48  # Hub z-position relative to cg [m]

        # Main rotor parameters
        self.omegareq = 44.4  # Required main rotor speed       [rad/s]
        self.R_mr = 4.912  # Main rotor radius               [m]
        self.c_mr = 0.27  # Main rotor blade chord          [m]
        self.Nb = 4  # Main rotor number of blades     [-]
        self.sigma_mr = self.Nb * self.c_mr / (np.pi * self.R_mr)  # Main rotor solidity         [-]
        self.Cla_mr = 6.113  # Main rotor liftgradient         [1/rad]
        self.delta0 = 0.0074  # MR profile drag coefficient     [-]
        self.delta2 = 38.66  # MR lift dependent prfl drag cf  [-]
        self.gamma_s = 0.0524  # MR shaft forward (pos) tilt     [rad]
        self.I_beta = 231.7  # Rotor blade flap mom of inertia [kg*m^2]
        self.twist = -0.14  # Rotor blade twist               [rad]
        self.e = 0.746  # Flapping hinge offset           [m]
        self.eps = self.e / self.R_mr  # Rotor blade offset (e/R)        [-]
        self.m_bl = 27.3  # Rotor blade mass                [kg]
        self.nu2 = 1.248  # Flap frequency ratio            [-]
        self.tau_lambda0_mr = 0.1  # Time constant                   [s]
        self.Kbeta = 113330  # Equivalent spring constant      [N*m/rad]
        self.gamma = self.rho0 * self.Cla_mr * self.c_mr * self.R_mr ** 4 / self.I_beta  # Lock nr at sea level [-]
        self.k = 1.15  # (van Holten p30)                [-]

        # Engine
        self.Ne = 2  # Number of engines                      [-]
        self.Pe = 313000  # Installed power per engine             [W]
        self.etam = 0.95  # Engine mechanical efficiency           [-]
        self.Kaeo = 0.8643  # Max power for AEO                      [-]
        self.Paeo = self.Ne * self.Pe * self.etam * self.Kaeo  # All engines operative available power  [W]
        self.Koei = 0.95  # Max power for OEI                      [-]
        self.Poei = self.Koei * (self.Ne - 1) * self.Pe * self.etam  # One engine inoperative available power [W]
        self.Koeitr = 1.10  # Max transient power for OEI            [-]
        self.Poeitr = self.Koeitr * (self.Ne - 1) * self.Pe * self.etam  # One engine inoperative transient avlbl pwr[W]
        self.Keng = -self.Pe * self.etam * self.Kaeo / self.omegareq / 0.02  # Gain between Omega and Peng [W/rad]

        # Fuselage
        self.F0 = 1.3 * 0.73  # Parasite drag area of the helicopter    [m^2]
        self.S_fus = self.F0 / 0.2  # Fuselage surface                        [m^2]
        self.K_fus = 0.83  # Correction coeff in fus pitching moment [-]
        self.Vol_fus = np.pi / 4 * 6 * 1.3  # Equivalent volume of circular body      [m^3]
        self.CDS = 1.2  # Eq. flat plate area (van Holten p52)    [m^2]
        self.n = 4.65  # (van Holten p51)                        [-]

        # Horizontal stabilizer
        self.Cla_hs = 4  # Horizontal stabilizer lift gradient       [1/rad]
        self.alpha0_hs = 0.0698  # Horizontal stabilizer incidence           [rad]
        self.S_hs = 0.803  # Horizontal stabilizer area                [m^2]
        self.x_hs = 4.64 - self.dxcg  # Horizontal stabilizer x-positon rel to cg [m]
        self.K_hs = 1.5  # Horizontal stabilizer downwash factor     [-]

        # Vertical fin
        self.Cla_fin = 4  # Vertical fin lift gradient             [1/rad]
        self.S_fin = 0.805  # Vertical fin area                      [m^2]
        self.beta0_fin = -0.06116  # Vertical fin incidence                 [rad]
        self.x_fin = 5.30 - self.dxcg  # Vertical fin x-position relative to cg [m]
        self.z_fin = 0.97  # Vertical fin z-position relative to cg [m]

        # Tail rotor
        self.R_tr = 0.95  # Tail rotor radius          [m]
        self.c_tr = 0.18  # Tail rotor blade chord     [m]
        self.f_tr = 1 - 3 * self.S_fin / (4 * self.R_tr ** 2 * np.pi)  # Tl rtr fin blockage factor [-]
        self.tail_rotor_gearing = 5.25  # Tail rotor gearing         [-]
        self.x_tr = 6.08 - self.dxcg  # Tail rotor x-pos rel to cg [m]
        self.z_tr = 1.72  # Tail rotor z-pos rel to cg [m]
        self.Cla_tr = 5.7  # Tail rotor lift gradient   [1/rad]
        self.Nb_tr = 2  # Tail rotor nr of blades    [-]
        self.sigma_tr = self.Nb_tr * self.c_tr / (np.pi * self.R_tr)  # Tail rotor solidity        [-]
        self.k_1_tr = 1  # MR downwash factor at TR   [-]
        self.tau_lambda0_tr = 0.3  # Time constant              [s]
        self.M_tr = 0.7  # Tail rotor figure of Merit [-]

        # Control system: from Garteur AG-06
        self.a1sl = np.deg2rad(-6)  # Lateral cyclic control range left             [deg]
        self.a1su = np.deg2rad(4)  # Lateral cyclic control range right            [deg]
        self.b1sl = np.deg2rad(10)  # Longitudinal cyclic control range back        [deg]
        self.b1su = np.deg2rad(-5.5)  # Longitudinal cyclic control range forward     [deg]
        self.t0l = np.deg2rad(2)  # Main rotor coll control range at 0.7*R_mr min [deg]
        self.t0u = np.deg2rad(18)  # Main rotor coll control range at 0.7*R_mr max [deg]
        self.t0trl = np.deg2rad(18)  # Tail rotor control range min                  [deg]
        self.t0tru = np.deg2rad(-6)  # Tail rotor control range max                  [deg]
        self.psipmu = np.deg2rad(-10)  # Control phase shift                           [deg]

        # Variables
        self.P_available = self.Paeo   # available engine power, switches based on engine status
        self.P_out = 0  # Output of engine integrator dynamics,
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.trim_controls = np.array([0, 0, 0, 0])

    def step(self, actions, virtual=False):

        state = self.integrate_runge_kutta(self.state, actions)
        if not virtual:
            self.state = state

        done = False
        if self.t > self.t_max:
            done = True
        reward = 0
        self.t += self.dt

        return state, reward, done

    def set_engine_status(self, n_engines_available=2, transient=False):
        if n_engines_available == 2:
            self.P_available = self.Paeo
        elif n_engines_available == 1 and not transient:
            self.P_available = self.Poei
        elif n_engines_available == 1 and transient:
            self.P_available = self.Poeitr
        elif n_engines_available == 0:
            self.P_available = 0
        else:
            raise ValueError("Incorrect input for setting the engine status")

    def calculate_state_derivatives(self, actions, state=None, trimming=True):

        #  Controls are fraction of blade angle between lower and upper bounds (e.g. 0 < u < 1)
        coll, long, lat, pedal = np.clip(actions, 0, 1.0)

        if state is None:
            u, v, w, p, q, r, phi, theta, psi, x, y, z, lambda0_mr, lambda0_tr, omega = self.state
        else:
            u, v, w, p, q, r, phi, theta, psi, x, y, z, lambda0_mr, lambda0_tr, omega = state

        # Converting stick positions to blade angles
        a1s2 = (self.a1su-self.a1sl)*lat + self.a1sl
        b1s2 = (self.b1su-self.b1sl)*long + self.b1sl

        # Control angles (in radians)
        theta_0 = (self.t0u-self.t0l)*coll + self.t0l            # Main rotor collective
        theta_1s = b1s2*np.cos(self.psipmu) + a1s2*np.sin(self.psipmu)  # Longitudinal cyclic
        theta_1c = a1s2*np.cos(self.psipmu) + b1s2*np.sin(self.psipmu)  # Lateral cyclic
        theta_0_tr = (self.t0tru-self.t0trl)*pedal + self.t0trl  # Tail rotor collective

        # Other variables
        rho = self.calculate_air_density(-z)  # Air density
        gamma = rho * self.Cla_mr * self.c_mr * self.R_mr**4 / self.I_beta  # Blade lock number
        alpha_cp = -np.arctan2(w, u) + theta_1s  # Angle of attack of control plane
        V = np.sqrt(u**2 + v**2 + w**2)  # Total airspeed

        omega_r = omega * self.R_mr
        mu_x = V*np.cos(alpha_cp)/omega_r   # Normalized x-velocity
        mu_z = -V*np.sin(alpha_cp)/omega_r  # Normalized z-velocity

        #######################
        # Main rotor dynamics #
        #######################
        eps = self.e / self.R_mr  # Rotor blade offset (e/R)        [-]
        den1 = gamma * (0.25 * eps**2 - eps/3 + 0.125)
        den2 = gamma * (mu_x**2 * (0.0625 * eps**2 - eps * 0.125 + 0.0625))

        A = np.eye(3)
        b = np.zeros((3, 1))
        A[0, 1] = gamma * mu_x * (2*eps**2 - eps) / (8*self.nu2)
        A[1, 2] = (1-self.nu2) / (den1 - den2)
        A[2, 0] = gamma * mu_x * (1/6 - 0.25*eps) / (-den1-den2)
        A[2, 1] = (1-self.nu2) / (-den1-den2)

        b[0, 0] = (gamma / (2*self.nu2)) * \
            (theta_0 * ((0.25 - eps/3) + mu_x**2*(0.25*eps**2-1/2*eps+0.25))
             + mu_x*(1/2*eps-1/3)*theta_1s
             + (mu_x**2/6 + 0.2 - mu_x**2*eps*0.25 - 0.25*eps)*self.twist
             - (1/3 - 0.5*eps)*(lambda0_mr-mu_z)
             + mu_x*(1/6 - 0.25*eps)*p/omega)
        b[1, 0] = (gamma *
                (mu_x*(2-3*eps)/6*theta_0
                 + (mu_x**2*(-3*eps**2+6*eps-3) + 16/6*eps - 2)/16*theta_1s
                 + mu_x/4*((1 - 4/3*eps)*self.twist - (eps**2-2*eps+1)*(lambda0_mr-mu_z))
                 + (1-4/3*eps)*p/(8*omega))-2*q/omega) / (den1-den2)
        b[2, 0] = (gamma*((8/3*eps - 2 + mu_x**2*(-eps**2+2*eps-1))/16*theta_1c + (1-4/3*eps)*q/(8*omega))
                   + 2*p/omega) / (-den1-den2)

        # Solving Ax=b for x yields the flapping angles
        a0, a1, b1 = np.linalg.solve(A, b)[:, 0]
        Ct = (self.Cla_mr*self.sigma_mr/8) * \
             ((2/3 + mu_x**2) * theta_0 * 2
              + (2 * theta_1s + p / omega) * mu_x
              + (mu_z - lambda0_mr) * 2
              + (1 + mu_x**2) * self.twist)  # CHECKED; NO ERRORS WRT MATLAB FILE (numerically equal until last decimal)

        Cd = self.delta0 + self.delta2 * Ct**2
        T_mr = Ct * rho * omega_r**2 * np.pi * self.R_mr**2

        # This appears to be unused?
        # lambda_d = V * np.sin(alpha_cp - a1) / omega_r + lambda0_mr
        # Cq = self.sigma_mr * Cd * 0.125 * (1 + 4.7 * mu_x**2) + Ct * lambda_d
        # Q_mr = Cq * dimless * R_mr

        a1t1s = a1 - theta_1s + self.gamma_s
        b1t1c = b1 + theta_1c

        # Pitch and roll moment due to eccentricity
        Le = omega_r**2 * eps * self.m_bl * np.sin(b1t1c)
        Me = omega_r**2 * eps * self.m_bl * np.sin(a1t1s)

        # Main rotor thrust coefficients: blade element method (bem) and Glauert (gl)
        Ct_bem_mr = Ct
        Ct_gl_mr = 2 * lambda0_mr * np.sqrt(mu_x**2 + (lambda0_mr - mu_z)**2)

        #######################
        # Tail rotor dynamics #
        #######################

        # Tail rotor tip speed [m/s]
        omegar_tr = self.tail_rotor_gearing * omega * self.R_tr

        # Normalized tail rotor velocity along x- and z-axes
        mu_x_tr = np.sqrt(u**2 + (w + self.k_1_tr*lambda0_mr*omega_r + q*self.x_tr)**2) / omegar_tr
        mu_z_tr = -(v - self.x_tr*r + self.z_tr*p) / omegar_tr

        # Tail rotor thrust coefficients: blade element method (bem) and Glauert (gl)
        Ct_bem_tr = self.Cla_tr*self.sigma_tr * (theta_0_tr*(2 + 3*mu_x_tr**2) + 3*mu_z_tr - 6*lambda0_tr) / 12
        Ct_gl_tr = 2*lambda0_tr*np.sqrt(mu_x_tr**2 + (mu_z_tr-lambda0_tr)**2)      # Checked: equal in order 10e-15

        # Forces and moments caused by the TR:
        T_tr = Ct_bem_tr * rho * omegar_tr**2 * np.pi * self.R_tr**2
        Y_tr = T_tr * self.f_tr
        L_tr = Y_tr * self.z_tr
        N_tr = -Y_tr * self.x_tr

        ############
        # Fuselage #
        ############

        alpha_fus = np.arctan2(w, u)
        R_fus = 0.5 * rho * V**2 * self.F0

        X_fus = -R_fus * np.cos(alpha_fus)
        Z_fus = -R_fus * np.sin(alpha_fus)
        M_fus = rho * V**2 * self.K_fus * self.Vol_fus * alpha_fus

        #########################
        # Horizontal stabilizer #
        #########################

        w_hs = (w + q*self.x_hs)  # Local flow at hs
        alpha_hs = self.alpha0_hs + np.arctan2(w_hs, u)  # HS incidence[rad]
        V_hs = np.sqrt(u**2 + w_hs**2)  # HS  velocity  [m/s]

        Z_hs = -rho/2 * V_hs**2 * 0.65 * self.S_hs * self.Cla_hs * alpha_hs
        M_hs = Z_hs * self.x_hs

        # Vertical fin
        v_fin = v - r*self.x_fin + p*self.z_fin
        beta_fin = self.beta0_fin + np.arctan2(v_fin, u)   # VF incidence[rad]
        V_fin = np.sqrt(u**2 + v_fin**2)   # VF  velocity[m / s]

        Y_fin = -rho/2 * V_fin**2 * self.S_fin * self.Cla_fin * beta_fin
        L_fin = self.z_fin * Y_fin
        N_fin = -self.x_fin * Y_fin

        ######################
        # Power calculations #
        ######################
        P_par = self.CDS * rho * V**3 / 2  # Parasite drag [W]
        P_i = np.abs(self.k * T_mr * lambda0_mr * omega_r)  # Induced power [W]
        PpPd = self.sigma_mr*Cd*rho*omega_r**3*np.pi*self.R_mr**2*(1+self.n*mu_x**2)/8  # Total profile drag power [W]
        P_c = -w * self.W  # Climb power [W]
        P_tr = np.abs(T_tr / self.M_tr * np.sqrt(np.abs(T_tr / (2*rho*np.pi*self.R_tr**2))))  # Tail rotor power [W]
        P_req = P_par + P_i + PpPd + P_c + P_tr   # Total power [W]

        tau = 0.3
        P_eng = self.Keng*(omega-self.omegareq*1.02)
        if trimming:
            self.P_out = P_eng
        else:
            P_in = P_eng
            self.P_out += (P_in - self.P_out) / tau * self.dt
            P_eng = self.P_out

        # Enforce engine limits
        P_eng = np.clip(P_eng, 0, self.P_available)

        # Summing rotor forces and moments
        Xmr = -T_mr * np.sin(a1t1s) * np.cos(b1t1c)
        Ymr = T_mr * np.sin(b1t1c)
        Zmr = -T_mr * np.cos(a1t1s) * np.cos(b1t1c)

        Lmr = Ymr * self.zh - Zmr * self.yh + Le
        Mmr = -Xmr * self.zh - Zmr * self.xh + Me
        Nmr = P_eng / omega + Xmr * self.yh - Ymr * self.xh

        # Equations of motion
        # Summation of all forces(except weight components)
        X = Xmr + X_fus
        Y = Ymr + Y_tr + Y_fin
        Z = Zmr + Z_fus + Z_hs

        Fx = -self.W * np.sin(theta) + X
        Fy = self.W * np.cos(theta) * np.sin(phi) + Y
        Fz = self.W * np.cos(theta) * np.cos(phi) + Z

        # Summation of all moments
        L = Lmr + L_tr + L_fin
        M = Mmr + M_fus + M_hs
        N = Nmr + N_tr + N_fin

        # Accelerations in the body-axes
        udot = Fx / self.mass - q*w + r*v
        vdot = Fy / self.mass - r*u + p*w
        wdot = Fz / self.mass - p*v + q*u

        rdot = (N - (self.Iyy-self.Ixx)*p*q + self.Jxz*((L - (self.Izz-self.Iyy)*q*r + self.Jxz*p*q)/self.Ixx - r*q)) / (self.Izz - self.Jxz**2/self.Ixx)  # checked
        qdot = (M - (self.Ixx-self.Izz)*r*p - self.Jxz*(p**2-r**2)) / self.Iyy
        pdot = (L - (self.Izz-self.Iyy)*q*r + self.Jxz*(rdot+p*q)) / self.Ixx

        psidot = (q*np.sin(phi) + r*np.cos(phi)) / np.cos(theta)
        thetadot = q*np.cos(phi) - r*np.sin(phi)
        phidot = p + psidot*np.sin(theta)

        # Rotor rotational acceleration = Nb*I times 1.1 to include rotating transmission
        Omegadot = rdot + (P_eng - P_req) / omega / (1.1*self.Nb*self.I_beta)

        # Velocities in Earth-axes
        xdot = (u*np.cos(theta) + (v*np.sin(phi)+w*np.cos(phi))*np.sin(theta))*np.cos(psi) \
            - (v*np.cos(phi)-w*np.sin(phi))*np.sin(psi)
        ydot = (u*np.cos(theta) + (v*np.sin(phi)+w*np.cos(phi))*np.sin(theta))*np.sin(psi) \
            + (v*np.cos(phi)-w*np.sin(phi))*np.cos(psi)
        zdot = -u*np.sin(theta) + (v*np.sin(phi)+w*np.cos(phi))*np.cos(theta)

        lambda0_mrdot = (Ct_bem_mr-Ct_gl_mr)/self.tau_lambda0_mr
        lambda0_trdot = (Ct_bem_tr-Ct_gl_tr)/self.tau_lambda0_tr

        dots = np.array([udot, vdot, wdot, pdot, qdot, rdot, phidot, thetadot, psidot,
                        xdot, ydot, zdot, lambda0_mrdot, lambda0_trdot, Omegadot])
        return dots

    def integrate_runge_kutta(self, old_state, actions):

        f = self.calculate_state_derivatives(actions, old_state)
        k1 = self.dt * f

        states = old_state + k1/2
        f = self.calculate_state_derivatives(actions, states)
        k2 = self.dt * f

        states = old_state + k2/2
        f = self.calculate_state_derivatives(actions, states)
        k3 = self.dt * f

        states = old_state + k3
        f = self.calculate_state_derivatives(actions, states)
        k4 = self.dt * f

        new_state = old_state + (k1 + 2*k2 + 2*k3 + k4) / 6

        # phi, theta, psi = new_state[6:9]
        # phi = np.arctan2(np.sin(phi), np.cos(phi))  # Get value clipped between +-180 deg
        # theta = np.arctan2(np.sin(theta), np.cos(theta))  # Get value clipped between +-180 deg
        # psi = np.arctan2(np.sin(psi), np.cos(psi))  # Get value clipped between +-180 deg
        # new_state[6:9] = phi, theta, psi

        return new_state

    def trim(self, trim_speed, flight_path_angle, altitude):

        def trim_to_state(states, trimvar):
            states[0] = trimvar[0]
            states[1] = trimvar[1]
            states[2] = trimvar[2]
            states[7] = trimvar[3]
            states[6] = trimvar[4]
            states[12] = trimvar[5]
            states[13] = trimvar[6]
            states[14] = trimvar[7]
            coll = trimvar[8]
            long = trimvar[9]
            lat = trimvar[10]
            pedal = trimvar[11]
            actions = np.array([coll, long, lat, pedal])
            return states, actions

        # Initial guesses
        V = trim_speed
        rho = self.calculate_air_density(altitude)  # Density of air                 [kg/m^3]
        D = rho / 2 * V ** 2 * self.F0  # Drag                           [N]
        theta = np.arcsin(-D * np.cos(flight_path_angle) / self.W)  # Fuselage pitch angle   [rad]
        u = V * np.cos(theta)  # Airspeed along x-axis          [m/s]
        v = 0  # Airspeed along y-axis          [m/s]
        w = V * np.sin(theta)  # Airspeed along z-axis          [m/s]
        p = 0  # Roll rate                      [rad/s]
        q = 0  # Pitch rate                     [rad/s]
        r = 0  # Yaw rate                       [rad/s]
        psi = 0  # Helicopter yaw angle           [rad]
        phi = 0  # Helicopter roll angle          [rad]
        lambda0_mr = 0.05  # Norm unif induc downwash of MR [-]
        lambda0_tr = 0.05  # Norm unif induc downwash of TR [-]
        coll = 0.75  # MR collective position         [#]
        long = 0.50  # Longitudinal cyclic position   [#]
        lat = 0.5  # Lateral cyclic position        [#]
        pedal = 0.50  # TR collective position         [#]
        x = 0  # Position along Earth x-axis    [m]
        y = 0  # Position along Earth y-axis    [m]
        z = -altitude  # Position along Earth z-axis    [m]
        omega = self.omegareq  # Main rotor speed               [rad/s]
        states = np.array([u, v, w, p, q, r, phi, theta, psi, x, y, z, lambda0_mr, lambda0_tr, omega])
        trimvar = np.array([states[0],  # 1:a  u
                            states[1],  # 2:  v
                            states[2],  # 3:  w
                            states[7],  # 4:  theta
                            states[6],  # 5:  phi
                            states[12],  # 6:  lambda0_mr
                            states[13],  # 7:  lambda0_tr
                            states[14],  # 8:  Omega
                            coll,  # 9:  collective
                            long,  # 10: longitudinal cyclic
                            lat,  # 11: lateral cyclic
                            pedal])  # 12: pedal

        f = [1]
        nn = 0

        delta = 1e-11
        dfdx = np.zeros((len(trimvar), len(trimvar)))

        while max(np.abs(f)) > 1e-8:
            nn = nn + 1

            states, actions = trim_to_state(states, trimvar)
            dot = self.calculate_state_derivatives(actions=actions, state=states, trimming=True)
            f = np.hstack((dot[0:6], dot[9:]))
            f[6] -= V * np.cos(flight_path_angle)
            f[8] += V * np.sin(flight_path_angle)

            oldtrimvar = trimvar

            for i in range(len(trimvar)):
                perturb = np.zeros(len(trimvar))
                perturb[i] = delta
                trimvar = oldtrimvar + perturb
                states, actions = trim_to_state(states, trimvar)
                dot = self.calculate_state_derivatives(actions=actions, state=states, trimming=True)
                fnew = np.hstack((dot[0:6], dot[9:]))
                fnew[6] -= V * np.cos(flight_path_angle)
                fnew[8] += V * np.sin(flight_path_angle)

                dfdx[:, i] = (fnew - f) / delta

            inc = -np.linalg.inv(dfdx) @ f[:, None]
            trimvar += inc.ravel()

        trim_state, trim_controls = trim_to_state(states, trimvar)
        self.state = trim_state
        # Control values are percentages: divide by 100 to get actions
        self.trim_controls = trim_controls

        return trim_state, self.trim_controls

    @staticmethod
    def calculate_air_density(altitude):
        """
        Calculates the air density at a given altitude.
        :param altitude:
        :return:
        """
        # density   = rho0*(1+lambda*h/T0)^(-1*(g/(R*lambda)+1))
        return 1.225 * (1 - 0.0065 * altitude / 288.15)**(-(9.80665 / (287.05 * -0.0065) + 1))


def test_6dof():
    dt = 0.02
    env = Helicopter6DOF(dt=dt)
    state, trim_controls = env.trim(trim_speed=35*0.5144, flight_path_angle=0, altitude=100*0.3048)
    stats = []
    while env.t < 40:
        if 1.0 < env.t < 1.48:
            action = trim_controls.copy()
            action[1] += 0.05
        else:
            action = trim_controls
        state, _, _ = env.step(actions=action)

        stats.append({'t': env.t,
                      'u': state[0],
                      'v': state[1],
                      'w': state[2],
                      'p': state[3],
                      'q': state[4],
                      'r': state[5],
                      'phi': state[6],
                      'theta': state[7],
                      'psi': state[8],
                      'x': state[9],
                      'y': state[10],
                      'z': state[11],
                      'omega': state[14],
                      'coll': action[0],
                      'long': action[1],
                      'lat': action[2],
                      'ped': action[3]})

    stats = DataFrame(stats)
    stats_matlab = DataFrame(loadmat("data.mat")["states"])
    plt.plot(stats['t'], stats['q'], 'b', label='q')
    plt.plot(stats['t'], stats['p'], 'r', label='p')
    plt.plot(stats['t'], stats['r'], 'g', label='r')
    plt.plot(stats['t'], stats_matlab[3], 'r--', label='pm')
    plt.plot(stats['t'], stats_matlab[4], 'b--', label='qm')
    plt.plot(stats['t'], stats_matlab[5], 'g--', label='rm')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dt = 0.01
    env = Helicopter6DOF(t_max=1, dt=dt)
    sns.set()
    trim_speeds = np.arange(0, 40, 0.1)
    trim_settings = list(map(lambda v: (env.trim(trim_speed=v, flight_path_angle=0, altitude=0)[1]), trim_speeds))
    plt.plot(trim_speeds, trim_settings)
    plt.xlabel('Trim speed [m/s]')
    plt.ylabel('Control setting [-]')
    plt.legend(['col', 'lon', 'lat', 'ped'])
    plt.show()
