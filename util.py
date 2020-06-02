import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from matplotlib import pyplot as plt


class Logger:
    """
    Container for the state and weight history of a single training episode.
    """
    def __init__(self, params):
        self.state_history = []
        self.weight_history = {'t': [],
                               'col':
                                    {'nn': {
                                        'a': {'i': [], 'o': []},
                                        'c': {'i': [], 'o': []},
                                        'tc': {'i': [], 'o': []}},
                                     'rls': {'F': [], 'G': []}},
                               'lon':
                                    {'nn':
                                        {'a': {'i': [], 'o': []},
                                         'c': {'i': [], 'o': []},
                                         'tc': {'i': [], 'o': []}},
                                     'rls': {'F': [], 'G': []}}}
        self.nn_weight_history = None
        self.training_parameters = params
        self.finalized = False

    def log_states(self, t, observation, ref, actions, rewards, Pav, Peng):
        """
        Logs a single environment transition.
        :param t: Time in seconds
        :param observation: numpy array of observed state
        :param ref: reference to follow
        :param actions: numpy array of actions taken
        :param rewards: numpy array of rewards obtained
        :param Pav: Power available, taken directly from environment
        :param Peng: Maximum engine power, taken directly from env
        :return:
        """
        self.state_history.append({'t': t,
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
                                   'Pav': Pav,
                                   'Peng': np.clip(Peng, 0, Pav),
                                   'omega': observation[-1],
                                   'ref': ref,
                                   'col': actions[0],
                                   'lon': actions[1],
                                   'lat': actions[2],
                                   'ped': actions[3],
                                   'r1': rewards[0],
                                   'r2': rewards[1]})

    def log_weights(self, t, agents, rls_estimator):
        network_sequence = ['a', 'c', 'tc']
        layer_sequence = ['i', 'o']
        self.weight_history['t'].append(t)
        for agent in agents:
            for nn, network_name in zip(agent.networks, network_sequence):
                for layer, layer_name in zip(nn.parameters(), layer_sequence):
                    self.weight_history[agent.control_channel_str]['nn'][network_name][layer_name].append(
                        layer.detach().numpy().ravel().copy())
            tmp_F, tmp_G = agent.get_transition_matrices(rls_estimator)
            self.weight_history[agent.control_channel_str]['rls']['F'].append(tmp_F.detach().numpy().ravel().copy())
            self.weight_history[agent.control_channel_str]['rls']['G'].append(tmp_G.detach().numpy().ravel().copy())

    def finalize(self):
        """
        Transform logged data into more user-friendly Pandas DataFrames. Must be called before plotting is possible.
        """
        if self.finalized is True:
            raise ValueError("Attempting to call logger.finalize(), but log is already finalized. ")
        w = self.weight_history
        weights = {'col':
                   {'ci': pd.DataFrame(data=w['col']['nn']['c']['i'], index=w['t']),
                    'co': pd.DataFrame(data=w['col']['nn']['c']['o'], index=w['t']),
                    'ai': pd.DataFrame(data=w['col']['nn']['a']['i'], index=w['t']),
                    'ao': pd.DataFrame(data=w['col']['nn']['a']['o'], index=w['t']),
                    'dsda': pd.DataFrame(data=w['col']['rls']['G'], index=w['t']),
                    'dsds': pd.DataFrame(data=w['col']['rls']['F'], index=w['t'])},
                   'lon':
                   {'ci': pd.DataFrame(data=w['lon']['nn']['c']['i'], index=w['t']),
                    'co': pd.DataFrame(data=w['lon']['nn']['c']['o'], index=w['t']),
                    'ai': pd.DataFrame(data=w['lon']['nn']['a']['i'], index=w['t']),
                    'ao': pd.DataFrame(data=w['lon']['nn']['a']['o'], index=w['t']),
                    'dsda': pd.DataFrame(data=w['lon']['rls']['G'], index=w['t']),
                    'dsds': pd.DataFrame(data=w['lon']['rls']['F'], index=w['t'])
                    }}
        self.state_history = pd.DataFrame(self.state_history)
        self.weight_history = weights
        self.finalized = True

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class RefGenerator:
    def __init__(self, T, dt, A, u_ref, t_switch, filter_tau):
        self.T = T
        self.dt = dt
        self.A = A
        self.z_ref = 0
        self.u_ref = u_ref
        self.t_switch = t_switch
        self.filter = FirstOrderLag(time_constant=filter_tau)
        self.task = None
        self.int_error_u = 0

    def get_ref(self, obs, t):
        ref = np.nan * np.ones_like(obs)
        if self.task == "hover":
            theta_ref = 0
            z_ref = self.z_ref
        elif self.task == "train_lon":
            theta_ref = np.deg2rad(self.A * (np.sin(2 * np.pi * t / self.T)))
            z_ref = 0
        elif self.task == "train_col":
            theta_ref = np.deg2rad(3) if obs[0] > 1 else np.deg2rad(2.5) * obs[0]
            z_ref = max((self.z_ref - 1 * (t - self.t_switch)), -25)
        elif self.task == "velocity":
            z_ref = self.z_ref
            u_ref = self.filter(t)
            u_error = u_ref - obs[0]
            if round(t, 4) < 20:
                ku = -2
            else:
                ku = -3
            theta_ref = np.deg2rad(ku * u_error + -0.05 * self.int_error_u)
            self.int_error_u += u_error*self.dt
            ref[0] = u_ref
        elif self.task == "descent":
            if round(t, 4) == 12.93:
                self.filter.new_setpoint(t0=t, original=obs[0], setpoint=-obs[0])
                self.filter.tau = 2.5
            if round(t, 4) <= 12.92:
                zdot = 1.93
                theta_ref = np.deg2rad(-0.1)
                z_ref = self.z_ref + zdot * (t - self.t_switch)
            else:
                zdot = 0.5
                u_ref = self.filter(t)
                u_error = u_ref-obs[0]
                theta_ref = np.deg2rad(-3.5 * u_error + 0.1 * self.int_error_u)
                self.int_error_u += u_error * self.dt
                z_ref = -5 + zdot * (t - 12.92)
                ref[0] = u_ref
        else:
            raise NotImplementedError("Unknown task type")
        ref[7] = theta_ref
        ref[11] = z_ref

        return ref

    def set_task(self, task: str, t, obs, **kwargs):
        self.task = task
        self.int_error_u = 0
        if task == "train_col":
            self.z_ref = kwargs['z_start']
        elif task == "hover":
            self.z_ref = obs[11]
        elif task == "velocity":
            self.z_ref = kwargs['z_start']
            self.filter.new_setpoint(t0=t, original=obs[0], setpoint=kwargs['velocity_filter_target'])
        elif task == "descent":
            self.t_switch = kwargs["t_switch"]
            self.z_ref = obs[11]


def envelope_limits_reached(obs, limit_pitch=89, limit_roll=89):
    """
    Check if env has violated performance limits that are deemed unrecoverable.
    :param obs: Observation vector, numpy array
    :param limit_pitch: Pitch angle limit in degrees
    :param limit_roll: Roll angle limit in degrees
    :return: bool: if a limit has been violated  |   which: None or str, explanation of violation
    """

    limit_reached = False
    which = None
    if abs(np.rad2deg(obs[7])) > limit_pitch:
        limit_reached = True
        which = 'pitch angle > ' + str(limit_pitch) + ' deg'
    elif abs(np.rad2deg(obs[6])) > limit_roll:
        limit_reached = True
        which = 'roll angle > ' + str(limit_roll) + ' deg'

    return limit_reached, which


class FirstOrderLag:
    """
    First order lag filter for smoothly transitioning a single variable to a new setpoint.
    """
    def __init__(self, time_constant):
        self.t0 = None
        self.x0 = None
        self.setpoint = None
        self.tau = time_constant

    def __call__(self, t):
        return self.get_result(t)

    def get_result(self, t):
        return self.x0 + self.setpoint * (1-np.exp((-1 * (t - self.t0) / self.tau)))

    def new_setpoint(self, t0, original, setpoint):
        self.t0 = t0
        self.x0 = original
        self.setpoint = setpoint


