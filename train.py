# Standard library
import itertools
import time
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
from util import Logger, get_ref, envelope_limits_reached, plot_rls_weights, plot_neural_network_weights, plot_stats, FirstOrderLag


def train(env_params, ac_params, rls_params, pid_params, results_path, seed=0, weight_save_interval=10, return_logs=True,
          save_logs=False, save_weights=False,  save_agents=False, load_agents=True, agents_path="", plot_states=True, plot_nn_weights=False, plot_rls=False):

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
        excitation[j, 1] = -np.sin(2*np.pi * j / 50) * np.exp(-j/100)
    #     #excitation[j + 400, 1] = np.sin(2 * np.pi * j / 50) * 2
    excitation = np.deg2rad(excitation)

    # Flags
    excitation_phase = False
    done = False
    if load_agents:
        update_col = True
    else:
        update_col = False
    update_lon = True
    success = True
    rewards = np.zeros(2)
    int_error_u = 0
    step = 0
    z_ref_start = 0
    ref_lagger = FirstOrderLag(time_constant=5)
    ref_lagger.new_setpoint(t0=0, original=0, setpoint=30)
    while not done:

        if step == 1000:
            excitation_phase = False

        # if step == env_params['step_switch']:
        #     z_ref_start = logger.state_history[-1]['z']
        #     agent_lon.learning_rate_actor *= 0.1
        #     agent_lon.learning_rate_critic *= 0.1
        #     update_col = True

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
        # if env.t < 20:
        #     A = 10
        # elif 20 <= env.t < 40:
        #     A = 15
        # else:
        #     A = 20
        # next_ref = get_ref(obs=next_observation, t=env.t, t_switch=env_params['t_switch'], z_ref_start=z_ref_start, A=A)

        next_ref = np.nan * np.ones_like(next_observation)
        u_ref = ref_lagger.get_result(env.t)
        u_err = u_ref - next_observation[0]
        pitch_ref = np.deg2rad(-0.075 * u_err + -0.025 * int_error_u)
        int_error_u += u_err * env.dt
        next_ref[0] = u_ref
        next_ref[7] = pitch_ref
        next_ref[11] = 5

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
            # print("Save envelope limits reached, stopping simulation. Seed: " + str(seed))
            # print("Cause of violation: " + envelope_limits_reached(observation)[1])
            success = False
            done = True

        # Next step..
        observation = next_observation
        ref = next_ref
        step += 1

        if np.isnan(actions).any():
            # print("NaN encounted in actions at timestep", step, " -- ", actions, "Seed: " + str(seed))
            success = False
            done = True

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

    from params import env_params, ac_params_test, rls_params, pid_params

    results_path = "results/mar/5/"
    agents_path = "saved_models/mar/5/"
    env_params['step_switch'] = 0
    env_params['t_switch'] = 0
    training_logs, score = train(env_params=env_params,
                                 ac_params=ac_params_test,
                                 rls_params=rls_params,
                                 pid_params=pid_params,
                                 results_path=results_path,
                                 agents_path=agents_path,
                                 seed=44,
                                 return_logs=True,
                                 save_logs=False,
                                 save_weights=True,
                                 save_agents=False,
                                 plot_states=True,
                                 plot_nn_weights=False,
                                 plot_rls=False)
