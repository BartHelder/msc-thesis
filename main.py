from heli_simple import SimpleHelicopter
from controllers import sarsa
from HDP3 import HDPAgentNumpy
from tasks import SimpleTrackingTask
import numpy as np


if __name__ == "__main__":
    task = SimpleTrackingTask(period=40)
    env = SimpleHelicopter(tau=0.25, k_beta=0, task=task, name='poep')
    np.random.seed()
    ep_rewards = []

    agent = HDPAgentNumpy(learning_rate=0.2, run_number=0, weights_std=0.2)
    agent.train(env, n_updates=5)


    # print("Before training: %f" % agent.test(env, render=True))
    # print("Starting training phase...")
    #
    # for j in range(1, 101):
    #     agent = HDPAgentNumpy(run_number=j)
    #     r = agent.train(env, n_episodes=1, n_updates=5)
    #     ep_rewards.append(r)
    # print("Finished training, testing...")
    #
    # env2 = SimpleHelicopter(tau=0.25, k_beta=0, task=task)
    # agent.test(env2, max_steps=80, render=True)
