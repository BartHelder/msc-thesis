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

class SimpleHelicopter():

    def __init__(self, tau, k_beta, name):
        self.dt = 0.01
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
        self.theta = 0
        self.theta_threshold = 30 * np.pi / 180
        self.q_space = (np.arange(-6, 6, 0.01)) * np.pi / 180
        self.a1_space = (np.arange(-1, 1, 0.01)) * np.pi / 180

    def step(self, u_cyclic, q_ref):
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
        a1_dot = (-1 / self.tau) * (a1 + 16 * q / (self.gamma * self.omega))
        q_dot = -self.th_iy * (u_cyclic - a1)

        a1 = a1 + self.dt * a1_dot
        q = q + self.dt * q_dot

        q_discretized = self.q_space[np.searchsorted(self.q_space, q)]
        a1_discretized = self.a1_space[np.searchsorted(self.a1_space, a1)]
        self.theta += self.dt * q_discretized
        self.state = np.array([q_discretized, a1_discretized])

        done = self.theta < -self.theta_threshold or self.theta > self.theta_threshold
        reward = -(q - q_ref)**2 / 2

        return self.state, reward, done, {}

    def reset(self):
        """"
        State variabloes: q, al
        """
        self.state = np.array([0, 0])

        return self.state


def lqr_controller(state):

    K = np.array([-1, -0.1])
    action = -np.matmul(K, state)
    return action


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy function from a given action-value function and epsilon
    :param Q: Dictionary that maps state -> action values. Each entry is of length nA,
    :param epsilon: Probability to select a random action (float, 0<=epsilon<1)
    :param nA: number of actions possible in the environment
    :return: policy function that takes the observation of the environment as an argument and returns the action
    choice probabilities in the form of an np.array of length nA
    """
    def policy_function(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_function


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: on-policy TD control, finds the optimal epsilon-greedy policy
    :param env:
    :param num_episodes:
    :param discount_factor:
    :param alpha:
    :param epsilon:
    :return:
    """

    # The (final) action-value function, nested dict
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Episode statistics
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))

    # Policy-to-follow:
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # Run through the episodes
    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode+1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

        for t in itertools.count():
            # Perform action:
            next_state, reward, done, _ = env.step(action)

            # Based on results, pick the next action:
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

            # Update statistics from reward etc
            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward

            # TD update:
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break
            state = next_state
            action = next_action

    return Q, stats

#
def plot_stats(stats, gain):

    df = pd.DataFrame(stats)
    sns.set()
    ax = plt.figure(figsize=(10, 6))
    plt.plot(df['t'], df['q_ref'] * 180 / np.pi, label='q_ref')
    plt.plot(df['t'], df['y1'] * 180 / np.pi, label='q')
    plt.plot(df['t'], df['y2'] * 180 / np.pi, label='a_1')
    plt.plot(df['t'], df['u'] * 180 / np.pi, label='u')
    plt.xlabel('Time [s]')
    plt.ylabel('q [deg/s]  |  u [deg]')
    plt.title(env.name + ' | k_beta: ' + str(env.k_beta) + ' | tau: ' + str(env.tau) + '| K: ' + str(gain))
    plt.legend()
    plt.show()


if __name__ == '__main__':

    env = cartpole.CartPoleEnv()
    kb_teetered = 0
    tau_teetered = 0.05
    kb_hingeless = 460000
    tau_hingeless = 0.25

    TeeteredHeli = SimpleHelicopter(name='Teetered rotor', tau=tau_teetered, k_beta=kb_teetered)
    RigidHeli = SimpleHelicopter(name='Hingeless rotor', tau=tau_hingeless, k_beta=kb_hingeless)
    envs = [TeeteredHeli]



    for env in envs:
        stats = []
        for i_episode in range(1):
            state = env.reset()
            action = 0
            t_end = 30
            tick = env.dt
            q_ref = 0
            contoller_gain = -10
            action_space = np.arange(-5, 5.01, 0.1) * np.pi / 180

            for t in np.arange(0, t_end, tick):

                next_state, reward, done, _ = env.step(action, q_ref)
                q_ref = 3 * np.pi / 180 * np.sin(t * 2 * np.pi / 5)
                #next_action = ff_controller(q_ref)
                q_e = (q_ref - lqr_controller(state))
                next_action = contoller_gain * q_e
                next_action_discrete = action_space[(np.searchsorted(action_space, next_action))]

                # next_action = (q_ref - controller(env))

                stats.append({'t': t, 'q_ref': q_ref, 'u': action, 'y1': state[0], 'y2': state[1]})
                if t > 20 or done:
                    break
                state, action = next_state, next_action_discrete

            plot_stats(stats, contoller_gain)

