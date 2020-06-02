# Standard library
import itertools
import datetime
import os
import pickle

# Other
import torch
import numpy as np

# Custom made
from agents import DHPAgent
from model import RecursiveLeastSquares
from heli_models import Helicopter6DOF
from PID import LatPedPID, CollectivePID6DOF
from util import Logger, envelope_limits_reached, RefGenerator
from plotting import plot_rls_weights, plot_neural_network_weights, plot_stats


def train(mode: str, env_params: dict, ac_params: dict, rls_params: dict, pid_params: dict, results_path: str,
          seed=0, return_logs=True, save_logs=False, save_weights=False,  weight_save_interval: int = 10,
          save_agents=False, load_agents=False, agents_path="",
          plot_states=True, plot_nn_weights=False, plot_rls=False):

    """
    Trains the integrated IDHP agent in the 6DOF environment for a single episode.

    :param mode: str indicating what task the agent should perform: train, test_1, or test_2
    :param env_params: dict, relevant parameters for environment setup
    :param ac_params: dict, relevant parameters for actor-critic setup
    :param rls_params: dict, relevant parameters for RLS estimator setup
    :param pid_params: relevant parameters for PID setup
    :param results_path: Save path for the training logs
    :param seed: Random seed for initialization
    :param return_logs: Return the logs as function output?
    :param save_logs: Save the logs to file?
    :param save_weights: Save the weights in the logger? Useful for debugging
    :param weight_save_interval: Number of timesteps between saving the neural network weights in the logger
    :param save_agents: Save the trained agents to file after training?
    :param load_agents: Load pre-trained agents from file before starting the tasks?
    :param agents_path: Save or load path for trained agents.
    :param plot_states: Plot the states?
    :param plot_nn_weights: Plot neural network weights after training? (Warning: takes a while)
    :param plot_rls: Plot the RLS estimator gradients after training?

    :return: Can return various tuples, depending on above settings
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Environment
    env = Helicopter6DOF(dt=env_params['dt'],
                         t_max=env_params['t_max'])
    trim_state, trim_actions = env.trim(trim_speed=env_params['initial_velocity'],
                                        flight_path_angle=env_params['initial_flight_path_angle'],
                                        altitude=env_params['initial_altitude'])
    observation = trim_state.copy()
    ref = trim_state.copy()
    ref_generator = RefGenerator(T=10, dt=env_params["dt"], A=10, u_ref=0, t_switch=60, filter_tau=2.5)

    # Logging
    logger = Logger(params=ac_params)

    # Agents:
    agent_col = DHPAgent(**ac_params['col'])
    agent_lon = DHPAgent(**ac_params['lon'])
    if load_agents:
        agent_col.load(agents_path+"col.pt")
        agent_lon.load(agents_path+"lon.pt")
        with open(agents_path+"rls.pkl", 'rb') as f:
            rls_estimator = pickle.load(f)
    else:
        # incremental RLS estimator
        rls_estimator = RecursiveLeastSquares(**rls_params)
    agents = [agent_col, agent_lon]

    # Create controllers
    lateral_pid = LatPedPID(phi_trim=trim_state[6],
                            lat_trim=trim_actions[2],
                            pedal_trim=trim_actions[3],
                            dt=env_params["dt"],
                            gains_dict=pid_params)
    collective_pid = CollectivePID6DOF(col_trim=trim_actions[0],
                                       h_ref=env_params['initial_altitude'],
                                       dt=env_params['dt'],
                                       proportional_gain=pid_params['Kh'])

    # Excitation signal for the RLS estimator
    excitation = np.load('excitation.npy')

    # Flags
    excitation_phase = False if load_agents else True
    update_col = True if load_agents else False
    update_lon = True
    success = True
    rewards = np.zeros(2)

    def update_agent(n):
        """
        Shorthand to update a single numbered agent after a single transition.
        :param n: Index of agent to update, per list 'agents' (0=col, 1=lon)
        """
        rewards[n], dr_ds = agents[n].get_reward(next_observation, ref)
        F, G = agents[n].get_transition_matrices(rls_estimator)
        agents[n].update_networks(observation, next_observation, ref, next_ref, dr_ds, F, G)

    # Main loop
    for step in itertools.count():
        lateral_cyclic, pedal = lateral_pid(observation)

        # TODO: It would be much nicer if reference generation would be an internal thing in the environment I guess
        if mode == "train":
            if step == 0:
                ref_generator.set_task(task="train_lon", t=0, obs=observation, velocity_filter_target=0)
                ref = ref_generator.get_ref(observation, env.t)
            elif step == 1000:
                excitation_phase = False
            elif step == env_params['step_switch']:
                agent_lon.learning_rate_actor *= 0.1
                agent_lon.learning_rate_critic *= 0.1
                update_col = True
                ref_generator.set_task("train_col", t=env.t, obs=observation, z_start=observation[11])

            # Get ref, action, take action
            if step < env_params['step_switch']:
                actions = np.array([collective_pid(observation),
                                    trim_actions[1]-0.5 + agent_lon.get_action(observation, ref),
                                    lateral_cyclic,
                                    pedal])
            else:
                actions = np.array([trim_actions[0]-0.5 + agent_col.get_action(observation, ref),
                                    trim_actions[1]-0.5 + agent_lon.get_action(observation, ref),
                                    lateral_cyclic,
                                    pedal])
        elif mode == "test_1":
            if step == 0:
                ref_generator.set_task(task="hover", t=0, obs=observation)
                ref = ref_generator.get_ref(observation, env.t)

            elif step == 500:
                ref_generator.set_task("velocity",
                                       t=env.t,
                                       obs=observation,
                                       z_start=0,
                                       velocity_filter_target=25-observation[0])

            elif step == 2000:
                ref_generator.set_task("velocity",
                                       t=env.t,
                                       obs=observation,
                                       z_start=0,
                                       velocity_filter_target=0-observation[0])

            actions = np.array([trim_actions[0] - 0.5 + agent_col.get_action(observation, ref),
                                trim_actions[1] - 0.5 + agent_lon.get_action(observation, ref),
                                lateral_cyclic,
                                pedal])
        elif mode == "test_2":
            if step == 0:
                ref_generator.set_task(task="descent", t=0, t_switch=0,  obs=observation)
                ref = ref_generator.get_ref(observation, env.t)

            elif step == 1000:
                env.set_engine_status(n_engines_available=1, transient=True)
            actions = np.array([trim_actions[0] - 0.5 + agent_col.get_action(observation, ref),
                                trim_actions[1] - 0.5 + agent_lon.get_action(observation, ref),
                                lateral_cyclic,
                                pedal])
        else:
            raise NotImplementedError("Training mode unknown. ")

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
        rls_estimator.update(observation, actions[:2], next_observation)

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

        logger.log_states(env.t, observation, ref, actions, rewards, env.P_available, env.P_out)
        if save_weights and (step % weight_save_interval == 0):
            logger.log_weights(env.t, agents, rls_estimator)

        if envelope_limits_reached(observation)[0]:
            print("Save envelope limits reached, stopping simulation. Seed: " + str(seed))
            print("Cause of violation: " + envelope_limits_reached(observation)[1])
            success = False
            done = True

        if np.isnan(actions).any():
            print("NaN encounted in actions at timestep", step, " -- ", actions, "Seed: " + str(seed))
            success = False
            done = True

        if done or (mode == "test_2" and observation[11] > 0):
            break

        # Next step..
        observation = next_observation
        ref = next_ref
        step += 1

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
        rls_estimator.save(path=agents_path+"rls.pkl")

    # Visualization
    if plot_states:
        plot_stats(logger)

    if plot_nn_weights and save_weights:
        plot_neural_network_weights(logger, figsize=(8, 6), agent_name='col', title='Collective')
        plot_neural_network_weights(logger, figsize=(8, 6), agent_name='lon', title='Longitudinal Cyclic')
    elif plot_nn_weights and not save_weights:
        print("Called plot_nn_weights but no weights were saved (save_weights=False), skipping. ")

    if plot_rls and save_weights:
        plot_rls_weights(logger)
    elif plot_rls and not save_weights:
        print("Called plot_rls_weights but no weights were saved (save_weights=False), skipping. ")

    score = np.sqrt(-logger.state_history.iloc[5000:6000]['r2'].sum()/1000)

    if return_logs:
        return logger, score
    else:
        if success:
            return 1, score
        else:
            return 0, 0


if __name__ == "__main__":
    from params import env_params_train, env_params_test1, env_params_test2, ac_params_train, ac_params_test, \
        rls_params, pid_params

    seed = 101
    dt = datetime.datetime.now()
    results_path = f"results/{dt.month}/{dt.day}/{dt.hour}/{str(dt.minute).zfill(2)}/"
    agents_path = "saved_models/mar/11/"

    training_logs, score = train(mode="train",
                                 env_params=env_params_train,
                                 ac_params=ac_params_train,
                                 rls_params=rls_params,
                                 pid_params=pid_params,
                                 results_path=results_path,
                                 agents_path=agents_path,
                                 seed=seed,
                                 return_logs=True,
                                 save_logs=False,
                                 save_weights=True,
                                 save_agents=False,
                                 load_agents=False,
                                 plot_states=True,
                                 plot_nn_weights=True,
                                 plot_rls=True)

    # training_logs, score = train(mode="test_1",
    #                              env_params=env_params_test1,
    #                              ac_params=ac_params_test,
    #                              rls_params=rls_params,
    #                              pid_params=pid_params,
    #                              results_path=results_path,
    #                              agents_path=agents_path,
    #                              seed=1,
    #                              return_logs=True,
    #                              save_logs=True,
    #                              save_weights=False,
    #                              save_agents=False,
    #                              load_agents=True,
    #                              plot_states=True,
    #                              plot_nn_weights=False,
    #                              plot_rls=False)

    # training_logs, score = train(mode="test_2",
    #                              env_params=env_params_test2,
    #                              ac_params=ac_params_test,
    #                              rls_params=rls_params,
    #                              pid_params=pid_params,
    #                              results_path=results_path,
    #                              agents_path=agents_path,
    #                              seed=112,
    #                              return_logs=True,
    #                              save_logs=True,
    #                              save_weights=False,
    #                              save_agents=False,
    #                              load_agents=True,
    #                              plot_states=True,
    #                              plot_nn_weights=False,
    #                              plot_rls=False)
