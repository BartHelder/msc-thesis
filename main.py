import numpy as np
import tensorflow as tf
import multiprocessing as mp
import itertools
import json
import pandas as pd
from heli_models import Helicopter6DOF
from plotting import plot_neural_network_weights_2, plot_stats_6dof, plot_policy_function, plot_rls_stats
from agents import HDPAgent, DHPAgent, HDPCritic, TFActor6DOF
from PID import LatPedPID
from model import RecursiveLeastSquares


save_weights = False
tf.random.set_seed(1)
cfp = "config_6dof.json"

env = Helicopter6DOF()
trim_state, trim_actions = env.trim(trim_speed=20, flight_path_angle=0, altitude=0)

# Create controllers
ColAgent = DHPAgent(cfp, actor=TFActor6DOF, control_channel="collective")
LonAgent = DHPAgent(cfp, actor=TFActor6DOF, control_channel="cyclic_lon")
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

# Incremental RLS estimator
rls_kwargs = {'state_size': len(observation), 'action_size': 4, 'gamma': 1, 'covariance': 10**8, 'constant': False}
RLS = RecursiveLeastSquares(**rls_kwargs)
rls_stats = {'t': [0],
             'wa_col': [RLS.gradient_action()[:6, 0].ravel().copy()],
             'wa_cyc': [RLS.gradient_action()[:6, 1].ravel().copy()],
             'ws': [RLS.gradient_state().ravel().copy()]}

# Add excitation to inputs
excitation = np.zeros((1000, 4))
for j in range(400):
    excitation[j, 0] = -np.sin(np.pi * j/50) * 0.05
    excitation[j+400, 1] = np.sin(np.pi *j/50) * 0.05
excitation_phase = True

ref = np.nan * np.ones_like(observation)
step = 0
update_agent = [False, True]
while not done:

    # Get new reference
    if env.t < 20:
        A = 0
    elif 20 <= env.t < 70:
        A = 10
    else:
        A = 15

    qref = np.deg2rad(np.sin(2 * np.pi * env.t / 10) * A)

    ref[4] = qref
    ref[11] = 0

    # Augment state with tracking errors
    augmented_states = (ColAgent.augment_state(observation, reference=ref),
                        LonAgent.augment_state(observation, reference=ref))

    lateral_cyclic, pedal = LatPedController(observation)

    # Get actions from actors
    actions = [ColAgent.actor(augmented_states[0]).numpy().squeeze(),
               LonAgent.actor(augmented_states[1]).numpy().squeeze(),
               #trim_actions[1],
               lateral_cyclic,
               pedal]

    # Add excitations (RLS windup phase) and trim values
    if excitation_phase:
        actions += excitation[step]

    actions[:2] += (trim_actions[:2] - 0.5)
    actions = np.clip(actions, 0, 1)

    # Take step in the environment
    next_observation, _, done = env.step(actions)

    # Update RLS model
    RLS.update(state=observation, action=actions, next_state=next_observation)


    # Get rewards, update actor and critic networks
    for agent, count in zip(agents, itertools.count()):
        reward[count], dr_ds = agent.get_reward(next_observation, ref)
        next_augmented_state = agent.augment_state(next_observation, ref)
        if update_agent[count]:
            agent.update_networks(s1=augmented_states[count],
                                  s2=next_augmented_state,
                                  F=RLS.gradient_state(),
                                  G=RLS.gradient_action(),
                                  dr_ds=dr_ds
                                  )



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

    rls_stats['t'].append(env.t)
    rls_stats['ws'].append(RLS.gradient_state().ravel())
    rls_stats['wa_col'].append(RLS.gradient_action()[:12, 0].ravel())
    rls_stats['wa_cyc'].append(RLS.gradient_action()[:12, 1].ravel())

    if ((int(env.t / env.dt)) % 10 == 0) and save_weights:
        weight_stats['t'].append(env.t)
        weight_stats['wci'].append(LonAgent.critic.trainable_weights[0].numpy().ravel().copy())
        weight_stats['wco'].append(LonAgent.critic.trainable_weights[1].numpy().ravel().copy())
        weight_stats['wai'].append(LonAgent.actor.trainable_weights[0].numpy().ravel().copy())
        weight_stats['wao'].append(LonAgent.actor.trainable_weights[1].numpy().ravel().copy())

    if env.t > 90 or abs(observation[7]) > np.deg2rad(89) or abs(observation[6]) > np.deg2rad(89):
        done = True

    if 8 < env.t < 16:
        update_agent = [True, True]
    if env.t > 16:
        excitation_phase = False
        update_agent = [True, True]

    # Next step..
    observation = next_observation
    step += 1

stats = pd.DataFrame(stats)
if save_weights:
    weights = {'wci': pd.DataFrame(data=weight_stats['wci'], index=weight_stats['t']),
               'wco': pd.DataFrame(data=weight_stats['wco'], index=weight_stats['t']),
               'wai': pd.DataFrame(data=weight_stats['wai'], index=weight_stats['t']),
               'wao': pd.DataFrame(data=weight_stats['wao'], index=weight_stats['t'])}
    plot_neural_network_weights_2(weights)
plot_stats_6dof(stats)
plot_rls_stats(rls_stats)
