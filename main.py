from heli_simple import SimpleHelicopter
from HDP3 import HDPAgentNumpy
from tasks import SimpleTrackingTask
import numpy as np
import multiprocessing as mp
import itertools
import json
from plotting import plot_neural_network_weights
from controllers import sarsa


def do_one_trial(env, learning_rate, weights_std, path="tmp.json", n_episodes=100, save_weights=False, results_to_json=True, plot_stats=False):

    ep_rewards = []
    ep_weights = []
    for j in range(1, n_episodes+1):
        agent = HDPAgentNumpy(learning_rate=learning_rate, run_number=j, weights_std=weights_std)
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

    task = SimpleTrackingTask(period=40)
    env = SimpleHelicopter(tau=0.25, k_beta=0, task=task, name='poep')
    # env2 = SimpleHelicopter(tau=0.25, k_beta=46000, task=task, name='snel')
    np.random.seed()
    n_episodes = 100

    do_one_trial(env, 0.4, 0.1, n_episodes=1, results_to_json=False, plot_stats=True)







