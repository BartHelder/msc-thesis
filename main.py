import numpy as np
import tensorflow as tf
import multiprocessing as mp
import itertools
import json
import pandas as pd
from heli_models import Helicopter6DOF
from plotting import plot_neural_network_weights_2, plot_stats_6dof, plot_policy_function
from HDP_tf import Agent, TFActor6DOF
from PID import LatPedPID
from model import RecursiveLeastSquares



save_weights = True
tf.random.set_seed(222)
cfp = "config_6dof.json"

env = Helicopter6DOF()
trim_state, trim_actions = env.trim(trim_speed=20, flight_path_angle=0, altitude=0)

ColAgent = Agent(cfp, actor=TFActor6DOF, control_channel="collective", actor_kwargs={})
LonAgent = Agent(cfp, actor=TFActor6DOF, control_channel="cyclic_lon", actor_kwargs={})
LatPedController = LatPedPID(config_path=cfp,
                             phi_trim=trim_state[6],
                             lat_trim=trim_actions[2],
                             pedal_trim=trim_actions[3])
agents = (ColAgent, LonAgent)
stats = []
weight_stats = {'t': [],
                'wci': [],
                'wco': [],
                'wai': [],
                'wao': []}
reward = [None, None]
done = False
observation = trim_state.copy()

excitation = np.zeros((1000, 2))
for j in range(400):
    excitation[j, 0] = -np.sin(np.pi*j/50)
    excitation[j+400, 1] = np.sin(2*np.pi*j/50) * 2
excitation = np.deg2rad(excitation)
excitation_phase = True

rls_kwargs = {'state_size': len(observation), 'action_size': 4, 'gamma': 1, 'covariance': 10**8, 'constant': False}
RLS = RecursiveLeastSquares(**rls_kwargs)

while not done:

    # Get new reference


    # Augment state with tracking errors
    augmented_states = (ColAgent.augment_state(observation, reference=ref),
                        LonAgent.augment_state(observation, reference=ref))

    lateral_cyclic, pedal = LatPedController(observation)
    # Get actions from actors
    actions = [ColAgent.actor(augmented_states[0]).numpy().squeeze(),
               LonAgent.actor(augmented_states[1]).numpy().squeeze(),
               lateral_cyclic,
               pedal]


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
        reward[count] = agent.get_reward(next_observation, ref)
        next_augmented_state = agent.augment_state(next_observation, ref)
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
                  'ref': ref.copy(),
                  'col': actions[0],
                  'lon': actions[1],
                  'lat': actions[2],
                  'ped': actions[3],
                  'r1': reward[0],
                  'r2': reward[1]})

    if ((int(env.t / env.dt)) % 10 == 0) and save_weights:
        weight_stats['t'].append(env.t)
        weight_stats['wci'].append(ColAgent.critic.trainable_weights[0].numpy().ravel().copy())
        weight_stats['wco'].append(ColAgent.critic.trainable_weights[1].numpy().ravel().copy())
        weight_stats['wai'].append(ColAgent.actor.trainable_weights[0].numpy().ravel().copy())
        weight_stats['wao'].append(ColAgent.actor.trainable_weights[1].numpy().ravel().copy())

    if env.t > 120 or abs(observation[7]) > np.deg2rad(100) or abs(observation[6]) > np.deg2rad(100):
        done = True

    # Next step..
    observation = next_observation

stats = pd.DataFrame(stats)
weights = {'wci': pd.DataFrame(data=weight_stats['wci'], index=weight_stats['t']),
           'wco': pd.DataFrame(data=weight_stats['wco'], index=weight_stats['t']),
           'wai': pd.DataFrame(data=weight_stats['wai'], index=weight_stats['t']),
           'wao': pd.DataFrame(data=weight_stats['wao'], index=weight_stats['t'])}
plot_neural_network_weights_2(weights)
plot_stats_6dof(stats)
