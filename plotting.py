import matplotlib as mpl
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
import seaborn as sns
import os
import json

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
FIGSIZE = (8, 6)


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


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

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close()
    else:
        plt.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close()
    else:
        plt.show()

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close()
    else:
        plt.show()

    return fig1, fig2, fig3


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

    sns.set(context='paper')
    if color_palette is None:
        cp = sns.color_palette()
    else:
        cp = sns.color_palette(color_palette)

    refs = np.stack(df['reference'])

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
        ax.plot(df['t'], np.rad2deg(refs[:, 5]), c=cp[3], ls='--')
        ax.plot(df['t'], df['q'] * 180 / np.pi, c=cp[0])
        ax.set_xticklabels([])
        plt.ylabel('q [deg/s]')
    else:
        ax.plot(df['t'], np.rad2deg(refs[:, 4]), c=cp[3], ls='--')
        ax.plot(df['t'], df['theta'] * 180 / np.pi, c=cp[0])
        ax.set_xticklabels([])
        plt.ylabel('theta [deg]')
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
            ax.plot(df['t'], np.rad2deg(refs[:, 5]), c=cp[3], ls='--')
            ax.plot(df['t'], df['q'] * 180 / np.pi, c=cp[0])
            ax.set_xticklabels([])
            plt.ylabel('q [deg/s]')
        else:
            ax.plot(df['t'], np.rad2deg(refs[:, 4]), c=cp[3], ls='--')
            ax.plot(df['t'], df['theta'] * 180 / np.pi, c=cp[0])
            ax.set_xticklabels([])
            plt.ylabel('theta [deg]')
        plt.xlabel('Time [s]')

        plt.show()


def plot_stats_6dof(df: pd.DataFrame, results_only=False, color_palette=None):

    sns.set(context='paper')
    if color_palette is None:
        cp = sns.color_palette()
    else:
        cp = sns.color_palette(color_palette)

    refs = np.stack(df['ref'])

    fig1 = plt.figure(figsize=(6, 6))
    ax = fig1.add_subplot(411)
    ax.plot(df['t'], -refs[:, 11], c=cp[3], ls='--', label='reference')
    ax.plot(df['t'], -df['z'], c=cp[0])
    plt.ylabel('h [m]')
    plt.legend()
    ax.set_xticklabels([])
    ax = fig1.add_subplot(412)
    ax.plot(df['t'], df['r1'], c=cp[0])
    plt.ylabel('Reward [-]')
    ax.set_xticklabels([])
    ax = fig1.add_subplot(413)
    ax.plot(df['t'], np.rad2deg(refs[:, 4]), c=cp[3], ls='--')
    ax.plot(df['t'], df['q'] * 180 / np.pi, c=cp[0])
    ax.set_xticklabels([])
    plt.ylabel('q [deg/s]')
    ax = fig1.add_subplot(414)
    ax.plot(df['t'], df['r2'], c=cp[0])
    plt.ylabel('Reward [-]')
    plt.xlabel('Time [s]')
    plt.show()

    # fig2 = plt.figure(figsize=(6, 6))
    # ax = fig2.add_subplot(411)
    # ax.plot(df['t'], df['col'], c=cp[0])
    # plt.ylabel('Collective [-]')
    # ax.set_xticklabels([])
    # ax = fig2.add_subplot(412)
    # ax.plot(df['t'], df['lon'], c=cp[0], label='')
    # plt.ylabel('Longitudinal [-]')
    # ax.set_xticklabels([])
    # ax = fig2.add_subplot(413)
    # ax.plot(df['t'], df['lat'], c=cp[0], label='')
    # plt.ylabel('Lateral [-]')
    # ax.set_xticklabels([])
    # ax = fig2.add_subplot(414)
    # ax.plot(df['t'], df['ped'], c=cp[0], label='')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Pedal [-]')
    # plt.show()

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
    ax.plot(df['t'], df['p']*180/np.pi, c=cp[0], label='p')
    ax.plot(df['t'], df['q']*180/np.pi, c=cp[1], label='q')
    ax.plot(df['t'], df['r']*180/np.pi, c=cp[2], label='r')
    plt.ylabel('Angular rates [deg/s]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.show()

    if not results_only:

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
        ax.plot(df['t'], refs[:, 6]*180/np.pi, ls='--', c=cp[3], label='reference')
        ax.plot(df['t'], refs[:, 7]*180/np.pi, ls='--', c=cp[3])
        ax.plot(df['t'], refs[:, 8]*180/np.pi, ls='--', c=cp[3])
        ax.plot(df['t'], df['phi']*180/np.pi, c=cp[0], label=r'$\phi$')
        ax.plot(df['t'], df['theta']*180/np.pi, c=cp[1], label=r'$\theta$')
        ax.plot(df['t'], df['psi']*180/np.pi, c=cp[2], label=r'$\psi$')
        plt.ylabel('Body angles [deg]')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.show()

        # fig3 = plt.figure(figsize=(6, 10))
        # ax = fig3.add_subplot(511)
        # ax.plot(df['t'], refs[:, 9], c=cp[3], ls='--', label='reference')
        # ax.plot(df['t'], df['x'], c=cp[0])
        # plt.ylabel('x [m]')
        # ax.set_xticklabels([])
        # ax = fig3.add_subplot(512)
        # ax.plot(df['t'], -refs[:, 11], c=cp[3], ls='--', label='reference')
        # ax.plot(df['t'], -df['z'], c=cp[0])
        # plt.ylabel('h [m]')
        # ax.set_xticklabels([])
        # ax = fig3.add_subplot(513)
        # ax.plot(df['t'], refs[:, 0], c=cp[3], ls='--', label='reference')
        # ax.plot(df['t'], df['u'], c=cp[0], label='state')
        # plt.ylabel('u [m/s]')
        # plt.legend()
        # ax.set_xticklabels([])
        # ax = fig3.add_subplot(514)
        # ax.plot(df['t'], refs[:, 2], c=cp[3], ls='--', label='reference')
        # ax.plot(df['t'], df['w'], c=cp[0])
        # plt.ylabel('w [m/s]')
        # ax.set_xticklabels([])
        # ax = fig3.add_subplot(515)
        # ax.plot(df['t'], np.rad2deg(refs[:, 7]), c=cp[3], ls='--', label='reference')
        # ax.plot(df['t'], df['theta'] * 180 / np.pi, c=cp[0])
        # plt.ylabel('theta [deg]')
        # plt.xlabel('Time [s]')
        #
        # plt.show()
        #
        # fig4 = plt.figure(figsize=(6, 10))
        # ax = fig4.add_subplot(511)
        # ax.plot(df['t'], refs[:, 10], c=cp[3], ls='--', label='reference')
        # ax.plot(df['t'], df['y'], c=cp[0])
        # plt.ylabel('y [m]')
        # ax.set_xticklabels([])
        #
        # ax = fig4.add_subplot(512)
        # ax.plot(df['t'], refs[:, 6] * 180 / np.pi, c=cp[3], ls='--', label='reference')
        # ax.plot(df['t'], df['phi'] * 180 / np.pi, c=cp[0])
        # plt.ylabel('phi [deg]')
        # ax.set_xticklabels([])
        #
        # ax = fig4.add_subplot(513)
        # ax.plot(df['t'], refs[:, 8] * 180 / np.pi, c=cp[3], ls='--', label='reference')
        # ax.plot(df['t'], df['psi'] * 180 / np.pi, c=cp[0], label='state')
        # plt.ylabel('psi [deg]')
        # plt.legend()
        # ax.set_xticklabels([])
        #
        # ax = fig4.add_subplot(514)
        # ax.plot(df['t'], refs[:, 1], c=cp[3], ls='--', label='reference')
        # ax.plot(df['t'], df['v'], c=cp[0])
        # plt.ylabel('v [m/s]')
        # ax.set_xticklabels([])
        #
        # ax = fig4.add_subplot(515)
        # ax.plot(df['t'], np.rad2deg(refs[:, 5]), c=cp[3], ls='--', label='reference')
        # ax.plot(df['t'], df['r'] * 180 / np.pi, c=cp[0])
        # plt.ylabel('r [deg/s]')
        # plt.xlabel('Time [s]')
        #
        # plt.show()

def plot_neural_network_weights(data):
    sns.set()
    sns.set_context('paper')

    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=data['wci'], dashes=False, legend=False,
                 palette=sns.color_palette("hls", len(data['wci'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title('Critic weights - input to hidden')
    plt.show()

    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=data['wco'], dashes=False, legend=False,
                 palette=sns.color_palette("hls", len(data['wco'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title('Critic weights - hidden to output')
    plt.show()

    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=data['wai'], dashes=False, legend=False,
                 palette=sns.color_palette("hls", len(data['wai'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title('Actor weights - input to hidden')
    plt.show()

    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=data['wao'], dashes=False, legend=False,
                 palette=sns.color_palette("hls", len(data['wao'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title('Actor weights - hidden to output')
    plt.show()


def plot_neural_network_weights_2(data):

    sns.set()
    sns.set_context('paper')

    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=data['wci'], dashes=False, legend=False, palette=sns.color_palette("hls", len(data['wci'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title('Critic weights - input to hidden')
    plt.show()

    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=data['wco'], dashes=False, legend=False, palette=sns.color_palette("hls", len(data['wco'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title('Critic weights - hidden to output')
    plt.show()

    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=data['wai'], dashes=False, legend=False, palette=sns.color_palette("hls", len(data['wai'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title('Actor weights - input to hidden')
    plt.show()

    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=data['wao'], dashes=False, legend=False, palette=sns.color_palette("hls", len(data['wao'].columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title('Actor weights - hidden to output')
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


def plot_rls_stats(rls_stats):
    columns = ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y', 'z']
    scols = set(columns)
    dropcoll = scols.difference({'w', 'z'})
    dropcyc = scols.difference({'theta', 'q'})
    wa_col = pd.DataFrame(data=rls_stats['wa_col'], index=rls_stats['t'], columns=columns)
    wa_cyc = pd.DataFrame(data=rls_stats['wa_cyc'], index=rls_stats['t'], columns=columns)
    wa_col = wa_col.drop(columns=dropcoll)
    wa_cyc = wa_cyc.drop(columns=dropcyc)

    plt.figure()
    sns.lineplot(data=wa_col, dashes=False, legend='full', palette=sns.color_palette("hls", len(wa_col.columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Gradient size [-]')
    plt.title('iRLS Collective gradients')
    plt.show()

    plt.figure()
    sns.lineplot(data=wa_cyc, dashes=True, legend='full', palette=sns.color_palette("hls", len(wa_cyc.columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Gradient size [-]')
    plt.title('iRLS Cyclic gradients')
    plt.show()


if __name__ == "__main__":

    with open('rewards.json', 'r') as f:
        rewards = json.load(f)

    compare_runs(rewards)
