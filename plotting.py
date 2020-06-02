import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Qt5Agg')
FIGSIZE = (8, 6)


def plot_policy_function(actor, x_range, y_range,  title="Policy function"):
    """
    Plots the policy as a surface plot.
    """

    X, Y = np.meshgrid(x_range, y_range)
    sns.set()
    # Find value for all (x, y) coordinates
    Z = np.apply_along_axis(lambda a: actor(np.array([[a[0], a[1]]])).numpy().squeeze(), 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(np.rad2deg(X), np.rad2deg(Y), np.rad2deg(Z), rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-15, vmax=15)
        ax.set_xlabel('Tracked state')
        ax.set_ylabel('Tracking error')
        ax.set_zlabel('Cyclic pitch [deg]')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z, "{}".format(title))
    return Z


def get_title(info: dict) -> str:
    return 'Episode # ' + str(info['run_number']) + ' | k_beta=' + str(info['k_beta']) + ' | tau=' + str(info['tau'])


def plot_stats_1dof(df: pd.DataFrame, info, show_u=False):

    title = get_title(info)

    #  Tracking performance plot
    sns.set()
    fig1 = plt.figure(figsize=FIGSIZE)
    if show_u:
        plt.plot(df['t'], df['u'] * 180 / np.pi, 'b--', label='u')

    plt.plot(df['t'], df['q_ref'] * 180 / np.pi, 'r', label='q_ref')
    plt.plot(df['t'], df['q'] * 180 / np.pi, 'y', label='q')
    # plt.plot(df['t'], df['a1'] * 180 / np.pi,'g',  label='a_1')
    plt.xlabel('Time [s]')
    plt.ylabel('q [deg/s]  |  u [deg]')
    plt.title(title)
    plt.legend()
    plt.show()

    #  Reward over time plot
    fig2 = plt.figure(figsize=FIGSIZE)
    plt.plot(df['t'], df['r'])
    plt.xlabel('Time [s]')
    plt.ylabel('Reward [-]')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_stats_3dof(df: pd.DataFrame, pitch_rate=True, results_only=False, color_palette=None):
    """
    Plots the states of the 3DOF simulation
    :param df: pandas DataFrame containing results
    :param pitch_rate: Bool, whether the tracked state is q (True) or theta (False)
    :param results_only: If True, only plot tracking performance and not other states
    :param color_palette: custom color palette, string
    :return:
    """
    sns.set(context='paper')
    if color_palette is None:
        cp = sns.color_palette()
    else:
        cp = sns.color_palette(color_palette)

    refs = np.stack(df['reference'])

    def plot_q():
        ax.plot(df['t'], np.rad2deg(refs[:, 5]), c=cp[3], ls='--')
        ax.plot(df['t'], df['q'] * 180 / np.pi, c=cp[0])
        ax.set_xticklabels([])
        plt.ylabel('q [deg/s]')
    def plot_theta():
        ax.plot(df['t'], np.rad2deg(refs[:, 4]), c=cp[3], ls='--')
        ax.plot(df['t'], df['theta'] * 180 / np.pi, c=cp[0])
        ax.set_xticklabels([])
        plt.ylabel('theta [deg]')

    fig1 = plt.figure(figsize=(6, 6))
    ax = fig1.add_subplot(411)
    ax.plot(df['t'], -refs[:, 1], c=cp[3], ls='--', label='reference')
    ax.plot(df['t'], -df['z'], c=cp[0])
    plt.ylabel('h [m]')
    plt.legend()
    ax.set_xticklabels([])
    ax = fig1.add_subplot(412)
    ax.plot(df['t'], df['r1'], c=cp[0])
    plt.ylabel('Reward [-]')
    ax.set_xticklabels([])
    ax = fig1.add_subplot(413)
    if pitch_rate:
        plot_q()
    else:
        plot_theta()
    ax = fig1.add_subplot(414)
    ax.plot(df['t'], df['r2'], c=cp[0])
    plt.ylabel('Reward [-]')
    plt.xlabel('Time [s]')
    plt.show()

    fig2 = plt.figure(figsize=(6, 4))
    ax = fig2.add_subplot(211)
    ax.plot(df['t'], df['collective'] * 180 / np.pi, c=cp[0], label='collective')
    plt.ylabel('Collective [deg]')
    ax.set_xticklabels([])
    ax = fig2.add_subplot(212)
    ax.plot(df['t'], df['cyclic'] * 180 / np.pi, c=cp[0], label='cyclic')
    plt.xlabel('Time [s]')
    plt.ylabel('Cyclic pitch [deg]')
    plt.show()

    if not results_only:
        fig3 = plt.figure(figsize=(6, 10))
        ax = fig3.add_subplot(511)
        ax.plot(df['t'], refs[:, 0], c=cp[3], ls='--', label='reference')
        ax.plot(df['t'], df['x'], c=cp[0])
        plt.ylabel('x [m]')
        ax.set_xticklabels([])

        ax = fig3.add_subplot(512)
        ax.plot(df['t'], -refs[:, 1], c=cp[3], ls='--', label='reference')
        ax.plot(df['t'], -df['z'], c=cp[0])
        plt.ylabel('h [m]')
        ax.set_xticklabels([])

        ax = fig3.add_subplot(513)
        ax.plot(df['t'], refs[:, 2], c=cp[3], ls='--', label='reference')
        ax.plot(df['t'], df['u'], c=cp[0], label='state')
        plt.ylabel('u [m/s]')
        plt.legend()
        ax.set_xticklabels([])

        ax = fig3.add_subplot(514)
        ax.plot(df['t'], refs[:, 3], c=cp[3], ls='--', label='reference')
        ax.plot(df['t'], df['w'], c=cp[0])
        plt.ylabel('w [m/s]')
        ax.set_xticklabels([])

        ax = fig3.add_subplot(515)
        if not pitch_rate:
            plot_q()
        else:
            plot_q()
        plt.xlabel('Time [s]')

        plt.show()


def plot_sensitivity_analysis(confidence_interval=99):

    sns.set()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    all = []
    filelist = os.listdir('jsons/')
    for i in filelist:
        with open("jsons/" + i, 'r') as f:
            all.append(pd.DataFrame(json.load(f)))
    data = pd.concat(all)
    data.rename(columns={'sigma': 'stdev'}, inplace=True)
    palette = sns.color_palette('mako_r', data['stdev'].nunique())
    f = sns.lineplot(data=data, ax=ax, x='lr', y='er', hue='stdev', legend='full',
                     ci=confidence_interval, dashes=False, palette=palette)
    ax.set_yscale('symlog', subsy=[2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_yticks([-800, -700, -600, -500, -400, -300, -200, -100, -50])
    ax.set_yticklabels([-800, -700, -600, -500, -400, -300, -200, -100, -50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel('Learning rate [-]')
    plt.ylabel('Mean reward (log scale) [-]')
    plt.show()


def compare_runs(runs_dict):
    sns.set()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    df = pd.DataFrame(runs_dict)
    df = df[['1', '3', '2']]
    df.columns = ['a', 'b', '46000 (corr)']
    f = sns.violinplot(data=df)

    plt.xlabel('k_beta [-]')
    plt.ylabel('Episode rewards [-]')

    plt.show()


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
    ax.plot(df['t'], df['r1'], c=cp[3], ls='--', label='col')
    ax.plot(df['t'], df['r2'] * 10 ** 3, c=cp[0], label='lon')
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
    ax.plot(df['t'], df['col']*100, c=cp[0], label='col')
    ax.plot(df['t'], df['lon']*100, c=cp[1], label='lon')
    ax.plot(df['t'], df['lat']*100, c=cp[2], label='lat')
    ax.plot(df['t'], df['ped']*100, c=cp[3], label='ped')
    plt.ylabel('Control input [%]')
    ax.set_xticklabels([])
    ax.legend(loc='lower right')
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
        # ax.plot(df['t'], refs[:, 9], ls='--', c=cp[3], label='reference')
        # ax.plot(df['t'], refs[:, 10], ls='--', c=cp[3])
        # ax.plot(df['t'], refs[:, 11], ls='--', c=cp[3])
        ax.plot(df['t'], df['x'], c=cp[0], label='x')
        ax.plot(df['t'], df['y'], c=cp[1], label='y')
        ax.plot(df['t'], df['z'], c=cp[2], label='z')
        plt.ylabel('Position [m]')
        ax.set_xticklabels([])
        plt.legend()
    ax = fig2.add_subplot(512)
    ax.plot(df['t'], refs[:, 0], ls='--', c=cp[3], label='ref')
    ax.plot(df['t'], refs[:, 1], ls='--', c=cp[3])
    ax.plot(df['t'], refs[:, 2], ls='--', c=cp[3])
    ax.plot(df['t'], df['u'], c=cp[0], label='u')
    ax.plot(df['t'], df['v'], c=cp[1], label='v')
    ax.plot(df['t'], df['w'], c=cp[2], label='w')
    plt.ylabel('Body velocities [m/s]')
    ax.set_xticklabels([])
    plt.legend()
    ax = fig2.add_subplot(513)
    # ax.plot(df['t'], refs[:, 6] * 180 / np.pi, ls='--', c=cp[3], label='reference')
    # ax.plot(df['t'], refs[:, 7] * 180 / np.pi, ls='--', c=cp[3])
    # ax.plot(df['t'], refs[:, 8] * 180 / np.pi, ls='--', c=cp[3])
    ax.plot(df['t'], df['phi'] * 180 / np.pi, c=cp[0], label=r'$\phi$')
    ax.plot(df['t'], df['theta'] * 180 / np.pi, c=cp[1], label=r'$\theta$')
    ax.plot(df['t'], df['psi'] * 180 / np.pi, c=cp[2], label=r'$\psi$')
    plt.ylabel('Body angles [deg]')
    ax.set_xticklabels([])
    plt.legend()
    ax = fig2.add_subplot(514)
    ax.plot(df['t'], df['Pav']/1000, c=cp[3], ls='--', label=r'$P_{av}$')
    ax.plot(df['t'], df['Peng']/1000, c=cp[0], label=r'$P_{req}$')
    plt.ylabel("Power [kW]")
    ax.set_xticklabels([])
    plt.legend()
    ax = fig2.add_subplot(515)
    ax.plot(df['t'], np.ones_like(df['t']) * 102, c=cp[3], ls='--', label='rpm limits')
    ax.plot(df['t'], np.ones_like(df['t']) * 95, c=cp[3], ls='--')
    ax.plot(df['t'], df['omega'] / 44.4 * 100, c=cp[0], label=r'$\Omega$')
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


def plot_all_weights(df):

    F_col = df.weight_history['col']['dsds']
    F_col.columns = [r'$\frac{\partial w}{\partial w}$',
                     r'$\frac{\partial w}{\partial z}$',
                     r'$\frac{\partial z}{\partial w}$',
                     r'$\frac{\partial z}{\partial z}$']
    G_col = df.weight_history['col']['dsda']
    G_col.columns = [r'$\frac{\partial w}{\partial \delta_{col}}$',
                     r'$\frac{\partial z}{\partial \delta_{col}}$']

    F_lon = df.weight_history['lon']['dsds']
    F_lon.columns = [r'$\frac{\partial q}{\partial q}$',
                     r'$\frac{\partial q}{\partial \theta}$',
                     r'$\frac{\partial \theta}{\partial q}$',
                     r'$\frac{\partial \theta}{\partial \theta}$']
    G_lon = df.weight_history['lon']['dsda']
    G_lon.columns = [r'$\frac{\partial q}{\partial \delta_{lon}}}$',
                     r'$\frac{\partial \theta}{\partial \delta_{lon}}$']

    def plot_weights(F, G, name):
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(411)
        sns.lineplot(data=F, dashes=False, legend='full')
        plt.ylabel(r'$\hat{F} [-]$')
        ax.set_xticklabels([])

        ax = fig.add_subplot(412)
        sns.lineplot(data=G, dashes=False, legend='full')
        plt.ylabel(r'$\hat{G} [-]$')
        ax.set_xticklabels([])

        w = df.weight_history[name]

        dc = pd.concat([w['ci'], w['co']], axis=1)
        ax = fig.add_subplot(413)
        sns.lineplot(data=dc.iloc[0:1290:10], ci=None, dashes=False, legend=False)
        plt.ylabel('w_a [-]')
        ax.set_xticklabels([])

        da = pd.concat([w['ai'], w['ao']], axis=1)
        ax = fig.add_subplot(414)
        sns.lineplot(data=da.iloc[0:1290:10], ci=None, dashes=False, legend=False)
        plt.ylabel('w_c [-]')
        plt.xlabel('Time [s]')
        plt.show()

    plot_weights(F_lon, G_lon, 'lon')
    plot_weights(F_col, G_col, 'col')


def plot_hyperparameter_search(df):

    df['lr_act'] = (df['lr_act']).astype('int')
    df['lr_crit'] = (df['lr_crit']).astype('int')

    df = df.rename(columns={"lr_act": "$\eta_a$", "lr_crit": "$\eta_c$"})

    palette = sns.cubehelix_palette(light=.7, dark=0.3, n_colors=3)
    # palette = sns.color_palette('viridis', n_colors=3)
    ax = sns.relplot(data=df,
                     kind='line',
                     y='success',
                     x='gamma',
                     hue='tau',
                     size='std',
                     legend='full',
                     palette=palette,
                     row="$\eta_a$",
                     col="$\eta_c$",
                     height=2.5,
                     aspect=1)
    ax.set(ylabel='Success rate[-]', xlabel='Gamma [-]')
    plt.show()

    ax = sns.relplot(data=df,
                     kind='line',
                     y='rms',
                     x='gamma',
                     hue='tau',
                     size='std',
                     legend='full',
                     palette=palette,
                     row="$\eta_a$",
                     col="$\eta_c$",
                     height=2.5,
                     aspect=1)
    ax.set(ylabel='RMS of tracking error [deg]', xlabel='Gamma [-]')
    plt.show()


if __name__ == "__main__":

    path = '/home/bart/PycharmProjects/msc-thesis/results/final/'
    df = pd.read_pickle("/home/bart/PycharmProjects/msc-thesis/results/apr/10/test_1/high/16/log.pkl")
    sns.set(context='paper')
    plot_stats(df)
