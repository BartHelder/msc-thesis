import numpy as np
import tensorflow as tf
import multiprocessing as mp
import itertools
import json
import pandas as pd
from heli_models import Helicopter6DOF
from plotting import plot_neural_network_weights_2, plot_stats_6dof, plot_policy_function
from HDP_tf import Agent
from PID import LatPedPID

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

    tf.random.set_seed(42)
    cfp = "config_6dof.json"

    env = Helicopter6DOF()
    trim_state, trim_actions = env.trim(trim_speed=20, flight_path_angle=0, altitude=0)

    ColAgent = Agent(cfp, control_channel="collective", trim_value=trim_actions[0])
    ColAgent.ds_da = tf.constant(np.array([[-0.8], [0.8]]))
    LonAgent = Agent(cfp, control_channel="cyclic_lon", trim_value=trim_actions[1])
    LonAgent.ds_da = tf.constant(np.array([[0], [-0.08], [0.08]]))
    LatPedController = LatPedPID(config_path=cfp,
                                 phi_trim=trim_state[6],
                                 lat_trim=trim_actions[2],
                                 pedal_trim=trim_actions[3])
    agents = (ColAgent, LonAgent)
    stats = []
    reward = [None, None]
    done = False
    observation = trim_state.copy()
    while not done:

        # Get new reference

        # Augment state with tracking errors
        augmented_states = (ColAgent.augment_state(observation, reference=trim_state),
                            LonAgent.augment_state(observation, reference=trim_state))

        lateral_cyclic, pedal = LatPedController(observation)
        # Get actions from actors
        actions = [ColAgent.actor(augmented_states[0]).numpy().squeeze(),
                   LonAgent.actor(augmented_states[1]).numpy().squeeze(),
                   lateral_cyclic,
                   pedal]

        if 1.0 < env.t < 1.5:
            actions[1] += 0.025
        # actions = [trim_actions[0],
        #            trim_actions[1],
        #            trim_actions[2],
        #            trim_actions[3],
        #            ]

        # actions = [trim_actions[0],
        #            trim_actions[1],
        #            lateral_cyclic,
        #            pedal
        #            ]

        # Take step in the environment
        next_observation, _, done = env.step(actions)

        # Get rewards, update actor and critic networks
        for agent, count in zip(agents, itertools.count()):
            reward[count] = agent.get_reward(next_observation, trim_state)
            next_augmented_state = agent.augment_state(next_observation, trim_state)
            td_target = reward[count] + agent.gamma * agent.critic(next_augmented_state)
            agent.update_networks(td_target, augmented_states[count], n_updates=1)
            if count == 1:
                break

        # Log data
        stats.append({'t': env.t,
                      'u': observation[0],
                      'v': observation[1],
                      'w': observation[2],
                      'p': observation[3],
                      'q': observation[4],
                      'r': observation[5],
                      'phi': observation[6],
                      'theta': observation[7],
                      'psi': observation[8],
                      'x': observation[9],
                      'y': observation[10],
                      'z': observation[11],
                      'ref': trim_state,
                      'col': actions[0],
                      'lon': actions[1],
                      'lat': actions[2],
                      'ped': actions[3],
                      'r1': reward[0],
                      'r2': reward[1]})
        if env.t > 60 or abs(observation[7]) > np.deg2rad(100) or abs(observation[6]) > np.deg2rad(100):
            done = True

        # Next step..
        observation = next_observation

    stats = pd.DataFrame(stats)
    plot_stats_6dof(stats)


