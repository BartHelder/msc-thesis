from heli_simple import SimpleHelicopter
from controllers import sarsa
from HDP3 import HDPAgentNumpy
from tasks import SimpleTrackingTask
import numpy as np
import itertools
import multiprocessing as mp
import pandas as pd
import itertools
import json

learning_rates = [0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
sigmas = [0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
argslist = list(itertools.product(learning_rates, sigmas))
paths = [str(x) + '-' + str(y) for x, y in (argslist)]
num_sets = len(argslist)//4

def do_one_trial(env, learning_rate, sigma, path, n_episodes):
    ep_rewards = []
    for j in range(1, n_episodes+1):
        agent = HDPAgentNumpy(learning_rate=learning_rate, run_number=j, weights_std=sigma)
        rewards, _, info = agent.train(env, plotstats=False, n_updates=5)
        ep_rewards.append(rewards)

    results = {'lr': learning_rate, 'sigma': sigma, 'er': ep_rewards}
    with open('jsons/'+path+'.json', 'w') as fp:
        json.dump(results, fp)

if __name__ == "__main__":
    task = SimpleTrackingTask(period=40)
    env = SimpleHelicopter(tau=0.25, k_beta=0, task=task, name='poep')
    np.random.seed()
    trial_rewards = []
    trial_weights = []
    n_episodes = 100

    # Use 4 cores at a time for three sets to make training manageable
    for l in range(0, num_sets):
        jobs = []
        for i in range(0 + 4 * l, 4 + 4 * l):
            process = mp.Process(target=do_one_trial,
                                 args=(env, argslist[i][0], argslist[i][1], paths[i], n_episodes))
            jobs.append(process)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()

        print("Set %d/%d done " % (l + 1, num_sets))



    # for learning_rate, stdev in itertools.product(learning_rates, stdevs):
    #     ep_rewards = []
    #     ep_weights = []
    #     for j in range(1, n_episodes+1):
    #         agent = HDPAgentNumpy(learning_rate=learning_rate, run_number=j, weights_std=stdev)
    #         rewards, weights, info = agent.train(env, plotstats=False, n_updates=5)
    #         ep_rewards.append(rewards)
    #         ep_weights.append(weights)
    #     trial_rewards.append((np.mean(ep_rewards), np.std(ep_rewards), np.min(ep_rewards), np.max(ep_rewards)))
    #     trial_weights.append(ep_weights)

