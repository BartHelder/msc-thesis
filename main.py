from heli_simple import SimpleHelicopter
from controllers import sarsa
from HDP3 import HDPAgentNumpy
from tasks import SimpleTrackingTask
import numpy as np
import itertools
import pandas as pd

learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
stdevs = [0.05, 0.075, 0.1, 0.2, 0.3, 0.5]

if __name__ == "__main__":
    task = SimpleTrackingTask(period=40)
    env = SimpleHelicopter(tau=0.25, k_beta=0, task=task, name='poep')
    np.random.seed()
    trial_rewards = []
    trial_weights = []
    n_episodes = 100

    for learning_rate, stdev in itertools.product(learning_rates, stdevs):
        ep_rewards = []
        ep_weights = []
        for j in range(1, n_episodes+1):
            agent = HDPAgentNumpy(learning_rate=learning_rate, run_number=j, weights_std=stdev)
            rewards, weights, info = agent.train(env, plotstats=False, n_updates=5)
            ep_rewards.append(rewards)
            ep_weights.append(weights)
        trial_rewards.append((np.mean(ep_rewards), np.std(ep_rewards), np.min(ep_rewards), np.max(ep_rewards)))
        trial_weights.append(ep_weights)

    print("Finished training")
