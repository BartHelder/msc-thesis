# Standard library
import itertools
import time
import os

# Other
import torch
import numpy as np
import seaborn as sns

# Custom made
from agents import DHPAgent
from model import RecursiveLeastSquares
from heli_models import Helicopter6DOF
from PID import LatPedPID, CollectivePID6DOF
from util import Logger, get_ref, envelope_limits_reached, plot_rls_weights, plot_neural_network_weights, plot_stats


def train(env_params, ac_params, rls_params, path, seed=0, weight_save_interval=10, save_logs=False, save_weights=False,
          save_agents=False, plot_states=True, plot_nn_weights=False, plot_rls=False):

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Environment
    env = Helicopter6DOF(dt=env_params['dt'],
                         t_max=env_params['t_max'])
    trim_state, trim_actions = env.trim(trim_speed=env_params['initial_velocity'],
                                        flight_path_angle=env_params['initial_flight_path_angle'],
                                        altitude=env_params['initial_altitude'])
    observation = trim_state.copy()
    ref = get_ref(observation, env.t, env_params['t_switch'], 0, A=10)

    # incremental RLS estimator
    RLS = RecursiveLeastSquares(**rls_params)

    # Logging
    logger = Logger(params=ac_params)

    # Agents:
    agent_col = DHPAgent(**ac_params['col'])
    agent_lon = DHPAgent(**ac_params['lon'])
    agents = [agent_col, agent_lon]
    # Create controllers
    LatPedController = LatPedPID(config_path='config_6dof.json',
                                 phi_trim=trim_state[6],
                                 lat_trim=trim_actions[2],
                                 pedal_trim=trim_actions[3])
    ColController = CollectivePID6DOF(col_trim=trim_actions[0],
                                      h_ref=env_params['initial_altitude'],
                                      dt=env_params['dt'],
                                      proportional_gain=0.005)

    # Excitation signal for the RLS estimator
    excitation = np.zeros((1000, 2))
    # for j in range(400):
    #     #excitation[j, 1] = -np.sin(np.pi * j / 50)
    #     #excitation[j + 400, 1] = np.sin(2 * np.pi * j / 50) * 2
    excitation = np.deg2rad(excitation)

    # Flags
    excitation_phase = False
    done = False
    update_col = False
    update_lon = True

    rewards = np.zeros(2)
    t_start = time.time()
    step = 0
    z_ref_start = 0

    while not done:

        if step == env_params['step_switch']:
            z_ref_start = logger.state_history[-1]['z']
            agent_lon.learning_rate_actor *= 0.1
            agent_lon.learning_rate_critic *= 0.1
            update_col = True

        # Get ref, action, take action
        lateral_cyclic, pedal = LatPedController(observation)
        if step < env_params['step_switch']:
            actions = np.array([ColController(observation),
                                (trim_actions[1]-0.5) + agent_lon.get_action(observation, ref),
                               lateral_cyclic,
                               pedal])
        else:
            actions = np.array([(trim_actions[0]-0.5) + agent_col.get_action(observation, ref),
                                (trim_actions[1]-0.5) + agent_lon.get_action(observation, ref),
                                lateral_cyclic,
                                pedal])

        # Add excitations (RLS windup phase) and trim values
        if excitation_phase:
            actions += excitation[step]

        actions = np.clip(actions, 0, 1)

        # Take step in the environment
        next_observation, _, done = env.step(actions)
        if env.t < 20:
            A = 10
        elif 20 <= env.t < 40:
            A = 15
        else:
            A = 20
        next_ref = get_ref(obs=next_observation, t=env.t, t_switch=env_params['t_switch'], z_ref_start=z_ref_start, A=A)
        # Update RLS estimator,
        RLS.update(observation, actions[:2], next_observation)

        def update_agent(n):
            rewards[n], dr_ds = agents[n].get_reward(next_observation, ref)
            F, G = agents[n].get_transition_matrices(RLS)
            agents[n].update_networks(observation, next_observation, ref, next_ref, dr_ds, F, G)

        # Collective:
        if update_col:
            update_agent(0)
        else:
            rewards[0] = 0

        # Cyclic
        if update_lon:
            update_agent(1)
        else:
            rewards[1] = 0

        logger.log_states(env.t, observation, ref, actions, rewards)
        if save_weights and (step % weight_save_interval == 0 or step < 100):
            logger.log_weights(env.t, agents, RLS)

        if envelope_limits_reached(observation)[0]:
            print("Save envelope limits reached, stopping simulation. ")
            print("Cause of violation: " + envelope_limits_reached(observation)[1])
            done = True

        # Next step..
        observation = next_observation
        ref = next_ref
        step += 1

        if np.isnan(actions).any():
            print("NaN encounted in actions at timestep", step, " -- ", actions)
            done = True

    print("Training time: ", time.time()-t_start)
    logger.finalize()

    if not os.path.exists(path) and (save_logs or save_agents):
        os.mkdir(path)

    if save_logs:
        logger.save(path=path+"log.pkl")

    if save_agents:
        agent_col.save(path="saved_models/mar/4/col.pt")
        agent_lon.save(path="saved_models/mar/4/lon.pt")

    # Visualization
    sns.set(context='paper')
    if plot_states:
        plot_stats(logger)

    if plot_nn_weights and save_weights:
        plot_neural_network_weights(logger, figsize=(8, 6), agent_name='col', title='Collective')
        plot_neural_network_weights(logger, figsize=(8, 6), agent_name='lon', title='Longitudinal Cyclic')

    if plot_rls:
        plot_rls_weights(logger)

    return logger
