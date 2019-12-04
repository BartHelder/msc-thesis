from heli_models import Helicopter1DOF, Helicopter3DOF
from tasks import SimpleTrackingTask, HoverTask
import HDP3
import numpy as np
import multiprocessing as mp
import itertools
import json
from plotting import plot_neural_network_weights_2, plot_stats_3dof, plot_policy_function
from controllers import sarsa


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



    #  Plotting
    x_range = np.deg2rad(np.arange(-5, 5, 0.25))
    y_range = np.deg2rad(np.arange(-5, 5, 0.25))

    Z = plot_policy_function(agent, x_range, y_range)
    #plot_neural_network_weights_2(weights)
