import matplotlib
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
FIGSIZE = (10, 6)


def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


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


def plot_policy_function(agent, x_range, y_range, title="Policy function"):
    """
    Plots the policy as a surface plot.
    """

    X, Y = np.meshgrid(x_range, y_range)
    sns.set()
    # Find value for all (x, y) coordinates
    Z = np.apply_along_axis(lambda a: agent.action(np.array([a[0], a[1]])), 2, np.dstack([X, Y]))
    # Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    # Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(np.rad2deg(X), np.rad2deg(Y), np.rad2deg(Z), rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-15.0, vmax=15.0)
        ax.set_xlabel('q [deg/s]')
        ax.set_ylabel('q_err [deg/s]')
        ax.set_zlabel('Cyclic pitch [deg]')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z, "{}".format(title))
    return Z


def plot_Q_function(Q, env, title="Policy"):
    """
    Plots the resulting policy of a Q-function for a grid-based environment
    """
    P = np.zeros(env.nS)
    for key in Q:
        P[key] = np.argmax(Q[key])
    P.reshape(env.shape)

    def plot_surface(P, title):
        fig = plt.figure(figsize=(20,10))


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

def plot_stats_3dof(df: pd.DataFrame, info):

    #title = get_title(info)

    #  Tracking performance plot
    sns.set()

    plt.figure(figsize=FIGSIZE)
    plt.plot(df['t'], df['collective'] * 180/np.pi, 'b--', label='collective')
    plt.plot(df['t'], df['cyclic'] * 180/np.pi, 'r--', label='cyclic')
    plt.xlabel('Time [s]')
    plt.ylabel('Control deflection [deg]')
    plt.legend()
    plt.show()

    plt.figure(figsize=FIGSIZE)
    plt.plot(df['t'], df['q'] * 180 / np.pi, label='q [deg/s]')
    plt.plot(df['t'], df['qref'] * 180 / np.pi, '--', label='qref [deg/s]')
    #
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch rate [deg/s]')
    plt.legend()
    plt.show()

    plt.figure(figsize=FIGSIZE)
    plt.plot(df['t'], df['u'], label='u')
    plt.plot(df['t'], df['w'], label='w')
    plt.plot(df['t'], df['theta'] * 180 / np.pi, label='theta [deg]')
    plt.ylabel('Body velocities [m/s]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.show()


    plt.figure(figsize=FIGSIZE)
    #plt.plot(df['t'], df['x'], label='x [m]')
    plt.plot(df['t'], -df['z'], label='h [m]')
    plt.plot(df['t'], np.ones_like(df['t']) * 27, 'r--')
    plt.plot(df['t'], np.ones_like(df['t']) * 23, 'r--')
    plt.xlabel('Time [s]')
    plt.ylabel("Height [m]")
    plt.legend()
    plt.show()

    plt.figure(figsize=FIGSIZE)
    plt.plot(df['t'], df['r'], label='reward')
    plt.ylabel('Reward [-]')
    plt.xlabel('Time [s]')

    plt.legend()
    plt.show()


def plot_neural_network_weights(data, info):

    sns.set()
    sns.set_context('paper')
    title = 'title'

    w_critic = data.iloc[:, :42]
    w_actor = data.iloc[:, 42:]
    fig3 = plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=w_critic, dashes=False, legend=False, palette=sns.color_palette("hls", len(w_critic.columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title('Critic weights - ' + title)
    plt.show()

    fig4 = plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=w_actor, dashes=False, legend=False, palette=sns.color_palette("hls", len(w_actor.columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron weight [-]')
    plt.title('Actor weights - ' + title)
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


if __name__ == "__main__":

    with open('rewards.json', 'r') as f:
        rewards = json.load(f)

    compare_runs(rewards)
