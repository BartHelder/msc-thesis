# Standard library
import itertools
import time
import datetime
import os
import pickle

# Other
import torch
import numpy as np
import seaborn as sns

# Custom made
from agents import DHPAgent
from model import RecursiveLeastSquares
from heli_models import Helicopter6DOF
from PID import LatPedPID, CollectivePID6DOF
from util import Logger, envelope_limits_reached, plot_rls_weights, plot_neural_network_weights, plot_stats, RefGenerator


def train(mode, env_params, ac_params, rls_params, pid_params, results_path, seed=0, weight_save_interval=10, return_logs=True,
          save_logs=False, save_weights=False,  save_agents=False, load_agents=False, agents_path="", plot_states=True, plot_nn_weights=False, plot_rls=False):

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Environment
    env = Helicopter6DOF(dt=env_params['dt'],
                         t_max=env_params['t_max'])
    trim_state, trim_actions = env.trim(trim_speed=env_params['initial_velocity'],
                                        flight_path_angle=env_params['initial_flight_path_angle'],
                                        altitude=env_params['initial_altitude'])
    observation = trim_state.copy()
    ref_generator = RefGenerator(T=10, dt=env_params["dt"], A=10, u_ref=0, t_switch=60, filter_tau=2)

    # Logging
    logger = Logger(params=ac_params)

    # Agents:
    agent_col = DHPAgent(**ac_params['col'])
    agent_lon = DHPAgent(**ac_params['lon'])
    if load_agents:
        agent_col.load(agents_path+"col.pt")
        agent_lon.load(agents_path+"lon.pt")
        with open(agents_path+"rls.pkl", 'rb') as f:
            RLS = pickle.load(f)
    else:
        # incremental RLS estimator
        RLS = RecursiveLeastSquares(**rls_params)

    agents = [agent_col, agent_lon]
    # Create controllers
    LatPedController = LatPedPID(phi_trim=trim_state[6],
                                 lat_trim=trim_actions[2],
                                 pedal_trim=trim_actions[3],
                                 dt=env_params["dt"],
                                 gains_dict=pid_params)
    ColController = CollectivePID6DOF(col_trim=trim_actions[0],
                                      h_ref=env_params['initial_altitude'],
                                      dt=env_params['dt'],
                                      proportional_gain=pid_params['Kh'])

    # Excitation signal for the RLS estimator
    excitation = np.zeros((1000, 4))
    for j in range(400):
        excitation[j, 1] = np.sin(2*np.pi * j / 50) * np.exp(-j/100)
        excitation[j+400, 0] = np.sin(2*np.pi * j / 50) * np.exp(-j/100)
    excitation = np.deg2rad(excitation)

    # Flags
    excitation_phase = False if load_agents else True
    update_col = True if load_agents else False
    update_lon = True
    success = True
    rewards = np.zeros(2)

    for step in itertools.count():
        lateral_cyclic, pedal = LatPedController(observation)
        if mode == "train":
            if step == 0:
                ref_generator.set_task(task="train_lon", t=0, obs=observation, velocity_filter_target=0)
                ref = ref_generator.get_ref(observation, env.t)
            if step == 1000:
                excitation_phase = False
            if step == env_params['step_switch']:
                agent_lon.learning_rate_actor *= 0.1
                agent_lon.learning_rate_critic *= 0.1
                update_col = True
                ref_generator.set_task("train_col", t=env.t, obs=observation, z_start=observation[11])
            elif step == 2*env_params['step_switch']:
                agent_lon.learning_rate_actor *= 0.1
                agent_lon.learning_rate_critic *= 0.1
                agent_col.learning_rate_actor *= 0.1
                agent_col.learning_rate_critic *= 0.1
                ref_generator.set_task("velocity", t=env.t, obs=observation, z_start=observation[11], velocity_filter_target=25)

            # Get ref, action, take action
            if step < env_params['step_switch']:
                actions = np.array([ColController(observation),
                                    trim_actions[1]-0.5 + agent_lon.get_action(observation, ref),
                                   lateral_cyclic,
                                   pedal])
            else:
                actions = np.array([trim_actions[0]-0.5 +agent_col.get_action(observation, ref),
                                    trim_actions[1]-0.5 +agent_lon.get_action(observation, ref),
                                    lateral_cyclic,
                                    pedal])
        elif mode == "test_1":
            if step == 0:
                ref_generator.set_task(task="hover", t=0, obs=observation)
                ref = ref_generator.get_ref(observation, env.t)

            if step == 500:
                ref_generator.set_task("velocity", t=env.t, obs=observation, z_start=observation[11], velocity_filter_target=25-observation[0])

            if step == 3000:
                ref_generator.set_task("velocity", t=env.t, obs=observation, z_start=observation[11], velocity_filter_target=0-observation[0])

            actions = np.array([trim_actions[0] - 0.5 + agent_col.get_action(observation, ref),
                                trim_actions[1] - 0.5 + agent_lon.get_action(observation, ref),
                                lateral_cyclic,
                                pedal])
        elif mode == "test_2":
            if step == 0:
                ref_generator.set_task(task="descent", t=0, t_switch=0,  obs=observation)
                ref = ref_generator.get_ref(observation, env.t)

            actions = np.array([trim_actions[0] - 0.5 + agent_col.get_action(observation, ref),
                                trim_actions[1] - 0.5 + agent_lon.get_action(observation, ref),
                                lateral_cyclic,
                                pedal])

        # Add excitations (RLS windup phase) and trim values
        if excitation_phase:
            actions += excitation[step]

        actions = np.clip(actions, 0, 1)

        # Take step in the environment
        next_observation, _, done = env.step(actions)
        if env.t < 20:
            ref_generator.A = 10
        elif 20 <= env.t < 40:
            ref_generator.A = 15
        else:
            ref_generator.A = 20
        next_ref = ref_generator.get_ref(observation, env.t)

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
            print("Save envelope limits reached, stopping simulation. Seed: " + str(seed))
            print("Cause of violation: " + envelope_limits_reached(observation)[1])
            success = False
            done = True

        # Next step..
        observation = next_observation
        ref = next_ref
        step += 1

        if np.isnan(actions).any():
            print("NaN encounted in actions at timestep", step, " -- ", actions, "Seed: " + str(seed))
            success = False
            done = True

        if done or -observation[11] < 0:
            break

    # print("Training time: ", time.time()-t_start)
    logger.finalize()

    if save_logs:
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        logger.save(path=results_path+"log.pkl")

    if save_agents:
        if not os.path.exists(agents_path):
            os.mkdir(agents_path)
        agent_col.save(path=agents_path+"col.pt")
        agent_lon.save(path=agents_path+"lon.pt")
        RLS.save(path=agents_path+"rls.pkl")

    # if input("Save agent weights? Y/N   ") == ("Y" or "y"):
    #     agent_col.save(path="saved_models/mar/5/col.pt")
    #     agent_lon.save(path="saved_models/mar/5/lon.pt")

    # Visualization
    sns.set(context='paper')
    if plot_states:
        plot_stats(logger)
    #

    if plot_nn_weights and save_weights:
        plot_neural_network_weights(logger, figsize=(8, 6), agent_name='col', title='Collective')
        plot_neural_network_weights(logger, figsize=(8, 6), agent_name='lon', title='Longitudinal Cyclic')

    if plot_rls:
        plot_rls_weights(logger)

    score = np.sqrt(-logger.state_history.iloc[5000:6000]['r2'].sum()/1000)

    if return_logs:
        return logger, score
    else:
        if success:
            return 1, score
        else:
            return 0, 0


if __name__ == "__main__":

    from params import env_params_train, env_params_test, ac_params_train, ac_params_test, rls_params, pid_params

    results_path = "results/mar/11/" + str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute).zfill(2) + "/"
    agents_path = "saved_models/mar/11/"
    # training_logs, score = train(mode="train",
    #                              env_params=env_params_train,
    #                              ac_params=ac_params_train,
    #                              rls_params=rls_params,
    #                              pid_params=pid_params,
    #                              results_path=results_path,
    #                              agents_path=agents_path,
    #                              seed=112,
    #                              return_logs=True,
    #                              save_logs=True,
    #                              save_weights=True,
    #                              save_agents=False,
    #                              load_agents=False,
    #                              plot_states=True,
    #                              plot_nn_weights=False,
    #                              plot_rls=False)

    training_logs, score = train(mode="test_2",
                                 env_params=env_params_test,
                                 ac_params=ac_params_test,
                                 rls_params=rls_params,
                                 pid_params=pid_params,
                                 results_path=results_path,
                                 agents_path=agents_path,
                                 seed=112,
                                 return_logs=True,
                                 save_logs=True,
                                 save_weights=False,
                                 save_agents=False,
                                 load_agents=True,
                                 plot_states=True,
                                 plot_nn_weights=False,
                                 plot_rls=False)
