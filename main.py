from heli_simple import SimpleHelicopter
from controllers import sarsa



if __name__ == "__main__":

    pass

    # kb_teetered = 0
    # tau_teetered = 0.05
    # kb_hingeless = 460000
    # tau_hingeless = 0.25
    #
    # TeeteredHeli = SimpleHelicopter(name='Teetered rotor', tau=tau_teetered, k_beta=kb_teetered)
    # RigidHeli = SimpleHelicopter(name='Hingeless rotor', tau=tau_hingeless, k_beta=kb_hingeless)
    # env = TeeteredHeli
    #
    # Q, stats, final_ep = sarsa(env, 15000, 1.0, 0.5, 0.05)
    # plotting.plot_episode_stats(stats)
    # df = pd.DataFrame(final_ep)
    # df['t'] *= 0.01
    # plot_stats(df, 'final')

    # for env in envs:
    #     stats = []
    #     for i_episode in range(1):
    #         state = env.reset()
    #         action = 0
    #         t_end = 30
    #         tick = env.dt
    #         q_ref = 0
    #         contoller_gain = -5
    #         action_space = np.arange(-5, 5.01, 0.5) * np.pi / 180
    #
    #         for t in np.arange(0, t_end, tick):
    #
    #             next_state, reward, done, _ = env.step(action, q_ref)
    #             q_ref = 3 * np.pi / 180 * np.sin(t * 2 * np.pi / 5)
    #             #next_action = ff_controller(q_ref)
    #             q_e = (q_ref - lqr_controller(state))
    #             next_action = contoller_gain * q_e
    #             next_action_discrete = action_space[(np.searchsorted(action_space, next_action))]
    #
    #             # next_action = (q_ref - controller(env))
    #
    #             stats.append({'t': t, 'q_ref': q_ref, 'u': action, 'y1': state[0], 'y2': state[1]})
    #             if t > 20 or done:
    #                 break
    #             state, action = next_state, next_action_discrete
    #
    #         df = pd.DataFrame(stats)
    #         plot_stats(df, contoller_gain)