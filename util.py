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

    def log_states(self, t, observation, ref, actions, rewards):
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


def get_ref(obs, t, t_switch, z_ref_start, A):
    ref = np.nan * np.ones_like(obs)
    if t < t_switch:
        ref[11] = 0
        theta_ref = np.deg2rad(A * (np.sin(2 * np.pi * t / 6)))
        # if np.rad2deg(obs[7]) > 20:
        #     qref -= 2 * abs(obs[7]-np.deg2rad(20))
        # elif np.rad2deg(obs[7]) < -20:
        #     qref += 2 * abs(obs[7] + np.deg2rad(20))
    else:
        # qref = np.clip(-obs[7] * 0.5, -np.deg2rad(5), np.deg2rad(5))
        theta_ref = np.deg2rad(-1.5)
        ref[11] = (z_ref_start - 20 * np.sin((t - 60) / 10))
    ref[7] = theta_ref
    zref = 0
    #ref[11] = zref
    max_zref_deviation = 5

    return ref


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

    def get_result(self, t):
        return self.x0 + self.setpoint * (1-np.exp((-1 * (t - self.t0) / self.tau)))

    def new_setpoint(self, t0, original, setpoint):
        self.t0 = t0
        self.x0 = original
        self.setpoint = setpoint

def get_ref_2(obs, t, dt, int_error_u):
    ref = np.nan * np.ones_like(obs)
    u_ref = 30
    u_err = u_ref - obs[2]
    pitch_ref = np.deg2rad(-1 * u_err + -0.05 * int_error_u)
    q_ref = 2 * pitch_ref
    q = obs[4]
    q_err = q_ref - q
    int_error_u += u_err * dt
    ref[0] = u_ref
    ref[7] = pitch_ref
    ref[11] = 0
    return ref, int_error_u


def plot_stats(log, cp=sns.color_palette()):
    if log.finalized is False:
        raise ValueError("Logs not yet finalized, so plots cannot be made. Call .finalize() before plotting")
    df = pd.DataFrame(log.state_history)
    refs = np.stack(df['ref'])

    fig1 = plt.figure(figsize=(6, 6))
    ax = fig1.add_subplot(311)
    ax.plot(df['t'], -refs[:, 11], c=cp[3], ls='--', label='reference')
    ax.plot(df['t'], -df['z'], c=cp[0])
    plt.ylabel('h [m]')
    plt.legend()
    ax.set_xticklabels([])
    ax = fig1.add_subplot(312)
    ax.plot(df['t'], df['r1'], c=cp[3], ls='--', label='Collective')
    ax.plot(df['t'], df['r2'] * 10 ** 3, c=cp[0], label='Lon Cyclic')
    plt.legend()
    plt.ylabel('Reward [-]')
    ax.set_xticklabels([])
    ax = fig1.add_subplot(313)
    # ax.plot(df['t'], refs[:, 4]*180/np.pi, c=cp[3], ls='--', label='reference')
    # ax.plot(df['t'], df['q']*180/np.pi, c=cp[0])
    ax.plot(df['t'], refs[:, 7] * 180 / np.pi, c=cp[3], ls='--', label='reference')
    ax.plot(df['t'], df['theta'] * 180 / np.pi, c=cp[0], label=r'$\theta$')
    plt.legend()
    # plt.ylabel('q [deg/s]')
    plt.ylabel('theta [deg]')
    plt.xlabel('Time [s]')
    plt.show()

    fig2 = plt.figure(figsize=(6, 6))
    ax = fig2.add_subplot(211)
    ax.plot(df['t'], df['col'], c=cp[0], label='Collective')
    ax.plot(df['t'], df['lon'], c=cp[1], label='Longitudinal')
    ax.plot(df['t'], df['lat'], c=cp[2], label='Lateral')
    ax.plot(df['t'], df['ped'], c=cp[3], label='Pedal')
    plt.ylabel('Control input [-]')
    ax.set_xticklabels([])
    plt.legend()
    ax = fig2.add_subplot(212)
    ax.plot(df['t'], df['p'] * 180 / np.pi, c=cp[0], label='p')
    ax.plot(df['t'], df['q'] * 180 / np.pi, c=cp[1], label='q')
    ax.plot(df['t'], df['r'] * 180 / np.pi, c=cp[2], label='r')
    plt.ylabel('Angular rates [deg/s]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.show()

    fig3 = plt.figure(figsize=(6, 10))
    ax = fig3.add_subplot(311)
    ax.plot(df['t'], refs[:, 9], ls='--', c=cp[3], label='reference')
    ax.plot(df['t'], refs[:, 10], ls='--', c=cp[3])
    ax.plot(df['t'], refs[:, 11], ls='--', c=cp[3])
    ax.plot(df['t'], df['x'], c=cp[0], label='x')
    ax.plot(df['t'], df['y'], c=cp[1], label='y')
    ax.plot(df['t'], df['z'], c=cp[2], label='z')
    plt.ylabel('Position [m]')
    ax.set_xticklabels([])
    plt.legend()

    ax = fig3.add_subplot(312)
    ax.plot(df['t'], refs[:, 0], ls='--', c=cp[3], label='reference')
    ax.plot(df['t'], refs[:, 1], ls='--', c=cp[3])
    ax.plot(df['t'], refs[:, 2], ls='--', c=cp[3])
    ax.plot(df['t'], df['u'], c=cp[0], label='u')
    ax.plot(df['t'], df['v'], c=cp[1], label='v')
    ax.plot(df['t'], df['w'], c=cp[2], label='w')
    plt.ylabel('Body velocities [m/s]')
    ax.set_xticklabels([])
    plt.legend()

    ax = fig3.add_subplot(313)
    ax.plot(df['t'], refs[:, 6] * 180 / np.pi, ls='--', c=cp[3], label='reference')
    ax.plot(df['t'], refs[:, 7] * 180 / np.pi, ls='--', c=cp[3])
    ax.plot(df['t'], refs[:, 8] * 180 / np.pi, ls='--', c=cp[3])
    ax.plot(df['t'], df['phi'] * 180 / np.pi, c=cp[0], label=r'$\phi$')
    ax.plot(df['t'], df['theta'] * 180 / np.pi, c=cp[1], label=r'$\theta$')
    ax.plot(df['t'], df['psi'] * 180 / np.pi, c=cp[2], label=r'$\psi$')
    plt.ylabel('Body angles [deg]')
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
    wa_lon.columns = [r'$\frac{\partial \theta}{\partial \theta_{1s}}}$',
                   r'$\frac{\partial q}{\partial \theta_{1s}}$']

    plt.figure()
    sns.lineplot(data=wa_col, dashes=False, legend='full', palette=sns.color_palette("hls", len(wa_col.columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Gradient size [-]')
    plt.title('iRLS Collective gradients')
    plt.show()

    plt.figure()
    sns.lineplot(data=wa_lon, dashes=True, legend='full', palette=sns.color_palette("hls", len(wa_lon.columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Gradient size [-]')
    plt.title('iRLS Cyclic gradients')
    plt.show()