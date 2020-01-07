import numpy as np
import tensorflow as tf
import multiprocessing as mp
import itertools
import json
import pandas as pd
from heli_models import Helicopter6DOF
from plotting import plot_neural_network_weights_2, plot_stats_3dof, plot_policy_function
from HDP_tf import Agent

def do_one_trial(env,
                 agent,
                 learning_rate,
                 weights_std,
                 path="tmp.json",
                 n_episodes=100,
                 save_weights=False,
                 results_to_json=True,
                 plot_stats=False):

    ep_rewards = []
    ep_weights = []
    for j in range(1, n_episodes+1):
        agent = agent(learning_rate=learning_rate, run_number=j, weights_std=weights_std)
        rewards, weights, info = agent.train(env, plotstats=plot_stats, n_updates=5)
        ep_rewards.append(rewards)

        if save_weights:
            ep_weights.append(weights)
    results = {'lr': learning_rate, 'sigma': weights_std, 'er': ep_rewards}

    if results_to_json:
        with open('jsons/'+path+'.json', 'w') as fp:
            json.dump(results, fp)

    if save_weights:
        return ep_rewards, ep_weights


def multiprocess_tasks(env, learning_rates, sigmas, n_episodes=100, n_cores=4):

    # Get all combinations of training args + translate them into save paths
    args = list(itertools.product(learning_rates, sigmas))
    save_paths = [str(x) + '-' + str(y) for x, y in args]

    # Use 4 cores at a time for three sets to make training manageable
    num_sets = len(args) // n_cores
    for l in range(0, num_sets):
        jobs = []
        for i in range(0 + n_cores * l, n_cores + n_cores * l):
            process = mp.Process(target=do_one_trial,
                                 args=(env, args[i][0], args[i][1], save_paths[i], n_episodes))
            jobs.append(process)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()

        print("Set %d/%d done " % (l + 1, num_sets))


if __name__ == "__main__":

    tf.random.set_seed(167)
    cfp = "config.json"

    env = Helicopter6DOF()
    env.setup_from_config(task="sinusoid", config_path=cfp)

    CollectiveAgent = Agent(cfp, control_channel="collective")
    CollectiveAgent.ds_da = tf.constant(np.array([[-0.8], [1.0]]))
    CyclicAgent = Agent(cfp, control_channel="cyclic_lon")
    CyclicAgent.ds_da = tf.constant(np.array([[0], [-0.08], [0.08]]))

    agents = (CollectiveAgent, CyclicAgent)
    observation, trim_actions = env.reset(v_initial=20)
    stats = []
    reward = [None, None]
    weight_stats = {'t': [], 'wci': [], 'wco': [], 'wai': [], 'wao': []}

    done = False
    while not done:

        # Get new reference
        reference = env.get_ref()

        # Augment state with tracking errors
        augmented_states = (CollectiveAgent.augment_state(observation, reference),
                            CyclicAgent.augment_state(observation, reference))

        # Get actions from actors
        actions = (CollectiveAgent.actor(augmented_states[0]).numpy().squeeze(),
                   CyclicAgent.actor(augmented_states[1]).numpy().squeeze())

        # Take step in the environment
        next_observation, _, done = env.step(actions)

        # Get rewards, update actor and critic networks
        for agent, count in zip(agents, itertools.count()):
            reward[count] = agent.get_reward(next_observation, reference)
            next_augmented_state = agent.augment_state(next_observation, reference)
            td_target = reward[count] + agent.gamma * agent.critic(next_augmented_state)
            agent.update_networks(td_target, augmented_states[count], n_updates=1)
            if count == 1:
                break

        # Log data
        stats.append({'t': env.t,
                      'x': observation[0],
                      'z': observation[1],
                      'u': observation[2],
                      'w': observation[3],
                      'theta': observation[4],
                      'q': observation[5],
                      'reference': env.get_ref(),
                      'collective': actions[0],
                      'cyclic': actions[1],
                      'r1': reward[0],
                      'r2': reward[1]})

        if env.t > 280:
            done = True

        # Next step..
        observation = next_observation

    stats = pd.DataFrame(stats)
    plot_stats_3dof(stats)


