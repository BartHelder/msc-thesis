import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from matplotlib import pyplot as plt

class Logger:

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
        :return:
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


# def get_ref(obs, t, t_switch, z_ref_start, A):
#     ref = np.nan * np.ones_like(obs)
#     if t < t_switch:
#         z_ref = 0
#         theta_ref = np.deg2rad(A/1.72 * (np.sin(2 * np.pi * t / 10) + np.sin(2 * np.pi * t / 20)))
#     elif t_switch <= t < 120:
#         theta_ref = np.deg2rad(-1.5)
#         z_ref = (z_ref_start - 20 * np.sin(2*np.pi*(t - t_switch) / 30))
#     else:
#         theta_ref = np.deg2rad(-1 * (30-obs[0]) + -0.05 * int_error_u)
#     ref[7] = theta_ref
#     ref[11] = z_ref
#     return ref


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
            theta_ref = z_ref = 0
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
            theta_ref = np.deg2rad(-3 * u_error + -0.05 * self.int_error_u)
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
                theta_ref = np.deg2rad(-4 * u_error + 0.1 * self.int_error_u)
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
        elif task == "velocity":
            self.z_ref = kwargs['z_start']
            self.filter.new_setpoint(t0=t, original=obs[0], setpoint=kwargs['velocity_filter_target'])
        elif task == "descent":
            self.t_switch = kwargs["t_switch"]
            self.z_ref = obs[11]


def envelope_limits_reached(obs, limit_pitch=89, limit_roll=89):
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


def plot_stats(log, cp=sns.color_palette(), plot_xz=False):
    if log.finalized is False:
        raise ValueError("Logs not yet finalized, so plots cannot be made. Call .finalize() before plotting")
    df = pd.DataFrame(log.state_history)
    refs = np.stack(df['ref'])

    fig1 = plt.figure(figsize=(6, 10))
    ax = fig1.add_subplot(511)
    ax.plot(df['t'], -refs[:, 11], c=cp[3], ls='--', label='reference')
    ax.plot(df['t'], -df['z'], c=cp[0])
    plt.ylabel('h [m]')
    plt.legend()
    ax.set_xticklabels([])
    ax = fig1.add_subplot(512)
    ax.plot(df['t'], df['r1'], c=cp[3], ls='--', label='Collective')
    ax.plot(df['t'], df['r2'] * 10 ** 3, c=cp[0], label='Lon Cyclic')
    plt.legend()
    plt.ylabel('Reward [-]')
    ax.set_xticklabels([])
    ax = fig1.add_subplot(513)
    ax.plot(df['t'], refs[:, 7] * 180 / np.pi, c=cp[3], ls='--', label='reference')
    ax.plot(df['t'], df['theta'] * 180 / np.pi, c=cp[0], label=r'$\theta$')
    plt.legend()
    plt.ylabel('theta [deg]')
    ax.set_xticklabels([])
    ax = fig1.add_subplot(514)
    ax.plot(df['t'], df['col'], c=cp[0], label='Collective')
    ax.plot(df['t'], df['lon'], c=cp[1], label='Longitudinal')
    ax.plot(df['t'], df['lat'], c=cp[2], label='Lateral')
    ax.plot(df['t'], df['ped'], c=cp[3], label='Pedal')
    plt.ylabel('Control input [-]')
    ax.set_xticklabels([])
    plt.legend()
    ax = fig1.add_subplot(515)
    ax.plot(df['t'], df['p'] * 180 / np.pi, c=cp[0], label='p')
    ax.plot(df['t'], df['q'] * 180 / np.pi, c=cp[1], label='q')
    ax.plot(df['t'], df['r'] * 180 / np.pi, c=cp[2], label='r')
    plt.ylabel('Angular rates [deg/s]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.show()

    fig2 = plt.figure(figsize=(6, 10))
    ax = fig2.add_subplot(511)
    if plot_xz:
        ax.plot(df['x'], -df['z'])
        plt.ylabel('Height [m]')
        plt.xlabel('x [m]')
    else:
        ax.plot(df['t'], refs[:, 9], ls='--', c=cp[3], label='reference')
        ax.plot(df['t'], refs[:, 10], ls='--', c=cp[3])
        ax.plot(df['t'], refs[:, 11], ls='--', c=cp[3])
        ax.plot(df['t'], df['x'], c=cp[0], label='x')
        ax.plot(df['t'], df['y'], c=cp[1], label='y')
        ax.plot(df['t'], df['z'], c=cp[2], label='z')
        plt.ylabel('Position [m]')
        ax.set_xticklabels([])
        plt.legend()
    ax = fig2.add_subplot(512)
    ax.plot(df['t'], refs[:, 0], ls='--', c=cp[3], label='reference')
    ax.plot(df['t'], refs[:, 1], ls='--', c=cp[3])
    ax.plot(df['t'], refs[:, 2], ls='--', c=cp[3])
    ax.plot(df['t'], df['u'], c=cp[0], label='u')
    ax.plot(df['t'], df['v'], c=cp[1], label='v')
    ax.plot(df['t'], df['w'], c=cp[2], label='w')
    plt.ylabel('Body velocities [m/s]')
    ax.set_xticklabels([])
    plt.legend()
    ax = fig2.add_subplot(513)
    ax.plot(df['t'], refs[:, 6] * 180 / np.pi, ls='--', c=cp[3], label='reference')
    ax.plot(df['t'], refs[:, 7] * 180 / np.pi, ls='--', c=cp[3])
    ax.plot(df['t'], refs[:, 8] * 180 / np.pi, ls='--', c=cp[3])
    ax.plot(df['t'], df['phi'] * 180 / np.pi, c=cp[0], label=r'$\phi$')
    ax.plot(df['t'], df['theta'] * 180 / np.pi, c=cp[1], label=r'$\theta$')
    ax.plot(df['t'], df['psi'] * 180 / np.pi, c=cp[2], label=r'$\psi$')
    plt.ylabel('Body angles [deg]')
    ax.set_xticklabels([])
    plt.legend()
    ax = fig2.add_subplot(514)
    ax.plot(df['t'], df['Pav'], c=cp[3], ls='--', label='Pav')
    ax.plot(df['t'], df['Peng'], c=cp[0], label='Preq')
    plt.ylabel("Power [W]")
    ax.set_xticklabels([])
    plt.legend()
    ax = fig2.add_subplot(515)
    ax.plot(df['t'], np.ones_like(df['t']) * 102, c=cp[3], ls='--', label='rpm limits')
    ax.plot(df['t'], np.ones_like(df['t']) * 95, c=cp[3], ls='--')
    ax.plot(df['t'], df['omega'] / 44.4 * 100, c=cp[0], label='omega')
    plt.ylabel('Rotor speed [%]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.show()


def plot_neural_network_weights(log, agent_name, figsize=(8,6), title='Title'):

    w = log.weight_history[agent_name]
    sns.set_context('paper')

    plt.figure(figsize=figsize)
    sns.lineplot(data=w['ci'], dashes=False, legend=False,
                 palette=sns.color_palette("hls", len(w['ci'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title(title + ' | critic weights - input to hidden layer')
    plt.show()

    plt.figure(figsize=figsize)
    sns.lineplot(data=w['co'], dashes=False, legend=False,
                 palette=sns.color_palette("hls", len(w['co'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title(title + ' | critic weights - hidden layer to output')
    plt.show()

    plt.figure(figsize=figsize)
    sns.lineplot(data=w['ai'], dashes=False, legend=False,
                 palette=sns.color_palette("hls", len(w['ai'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title(title + ' | actor weights - input to hidden layer')
    plt.show()

    plt.figure(figsize=figsize)
    sns.lineplot(data=w['ao'], dashes=False, legend=False,
                 palette=sns.color_palette("hls", len(w['ao'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title(title + ' | actor weights - hidden layer to output')
    plt.show()


def plot_rls_weights(log):

    wa_col = log.weight_history['col']['dsda']
    wa_col.columns = [r'$\frac{\partial w}{\partial \theta_0}$',
                      r'$\frac{\partial z}{\partial \theta_0}$']

    wa_lon = log.weight_history['lon']['dsda']
    wa_lon.columns = [r'$\frac{\partial q}{\partial \theta_{1s}}}$',
                      r'$\frac{\partial \theta}{\partial \theta_{1s}}$']

    plt.figure()
    sns.lineplot(data=wa_col, dashes=False, legend='full', palette=sns.color_palette("hls", len(wa_col.columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Gradient size [-]')
    plt.title('iRLS Collective gradients')
    plt.show()

    plt.figure()
    sns.lineplot(data=wa_lon, dashes=False, legend='full', palette=sns.color_palette("hls", len(wa_lon.columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Gradient size [-]')
    plt.title('iRLS Cyclic gradients')
    plt.show()
