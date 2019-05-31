import numpy as np
import plotting
import sys
import itertools
import pandas as pd


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
    def policy_function(state):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_function


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, max_episode_length=20, start_Q=None):
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
    if start_Q is not None:
        Q = start_Q
    else:
        Q = np.zeros((len(env.q_space), len(env.qe_space), len(env.a1_space), len(env.action_space)))

    # Episode statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Policy-to-follow:
    policy = make_epsilon_greedy_policy(Q, epsilon, len(env.action_space))
    # Run through the episodes
    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode+1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        intermediate_stats = []
        for t in itertools.count():

            time = t * env.dt
            q_ref = 3 * np.pi / 180 * np.sin(time * 2 * np.pi / 5)

            # Perform action:
            next_state, reward, done, _ = env.step(action, q_ref)

            # Based on results, pick the next action:
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # Update statistics from reward etc
            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward

            # TD update:
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            if (i_episode+1) % 1000 == 0:
                intermediate_stats.append({'t': t,
                                           'q_ref': q_ref,
                                           'u': env.action_space[action],
                                           'q': env.state[0],
                                           'a1': env.state[1]})

            if done or t >= (max_episode_length / env.dt):
                break
            state = next_state
            action = next_action
        if len(intermediate_stats) > 0:
            df = pd.DataFrame(intermediate_stats)
            plotting.plot_stats(df, env, str(i_episode+1))

    return Q, stats, intermediate_stats

