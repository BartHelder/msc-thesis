import numpy as np
import pandas as pd
from collections import deque
from heli_models import Helicopter1DOF


class HDPAgentNumpy:

    def __init__(self,
                 discount_factor=0.95,
                 learning_rate=0.1,
                 run_number=0,
                 action_scaling=np.deg2rad(15),
                 scaling='std',
                 weights_std=0.1,
                 n_hidden=6,
                 n_inputs=3,
                 n_actions=1):
        self.gamma = discount_factor
        self.learning_rate = learning_rate
        self.action_scaling = action_scaling

        if scaling == 'std':
            self.w_critic_input_to_hidden = np.random.randn(n_inputs, n_hidden) * weights_std
            self.w_critic_hidden_to_output = np.random.randn(n_hidden, 1) * weights_std
            self.w_actor_input_to_hidden = np.random.randn(n_inputs, n_hidden) * weights_std
            self.w_actor_hidden_to_output = np.random.randn(n_hidden, n_actions) * weights_std

        elif scaling == 'xavier':
            self.w_critic_input_to_hidden = np.random.randn(n_inputs, n_hidden) * np.sqrt(2 / (n_inputs+n_hidden))
            self.w_critic_hidden_to_output = np.random.randn(n_hidden, 1) * np.sqrt(2 / (n_hidden+1))
            self.w_actor_input_to_hidden = np.random.randn(n_inputs, n_hidden) * np.sqrt(2 / (n_inputs+n_hidden))
            self.w_actor_hidden_to_output = np.random.randn(n_hidden, n_actions) * np.sqrt(2 / (n_hidden+n_actions))

        self.run_number = run_number

    def action(self, obs):

        a1 = np.matmul(obs[None, :], self.w_actor_input_to_hidden)
        h = np.tanh(a1)
        a2 = np.matmul(h, self.w_actor_hidden_to_output)

        return (self.action_scaling * np.tanh(a2)).squeeze()

    def train(self, env, plotstats=True, n_updates=5, anneal_learning_rate=False, annealing_rate=0.9994):

        # training loop: collect samples, send to optimizer, repeat updates times
        episode_rewards = [0.0]

        wci = self.w_critic_input_to_hidden
        wco = self.w_critic_hidden_to_output
        wai = self.w_actor_input_to_hidden
        wao = self.w_actor_hidden_to_output

        # This is a property of the environment
        dst_da = env.get_environment_transition_function()

        def _critic(obs):
            c_hidden_in = np.matmul(obs[None, :], wci)
            hidden = np.tanh(c_hidden_in)
            value = np.matmul(hidden, wco)

            return value.squeeze(), hidden

        def _actor(obs, scale=self.action_scaling):
            a1 = np.matmul(obs[None, :], self.w_actor_input_to_hidden)
            h = np.tanh(a1)
            a2 = np.matmul(h, self.w_actor_hidden_to_output)

            return (scale * np.tanh(a2)).squeeze(), h

        # Initialize environment and tracking task
        observation = env.reset()
        stats = []
        weight_stats = {'wci': [wci.ravel().copy()],
                        'wco': [wco.ravel().copy()],
                        'wai': [wai.ravel().copy()],
                        'wao': [wao.ravel().copy()]}
        info = {'run_number': self.run_number,
                'learning_rate': self.learning_rate,
                'stats': env.stats}

        # Repeat (for each step t of an episode)
        for step in range(int(env.episode_ticks)):

            # 1. Obtain action from critic network using current knowledge
            action, hidden_action = _actor(observation, scale=self.action_scaling)

            # 2. Obtain value estimate for current state
            # value, hidden_v = _critic(observation)

            # 3. Perform action, obtain next state and reward info
            next_observation, reward, done = env.step(action.squeeze())

            # TD target remains fixed per time-step to avoid oscillations
            td_target = reward + self.gamma * _critic(next_observation)[0]

            # Update models x times per timestep
            for j in range(n_updates):

                # 4. Update critic: error is td_target minus value estimate for current observation
                v, hc = _critic(observation)
                e_c = td_target - v

                #  dEc/dwc = dEc/dec * dec/dV * dV/dwc  = standard backprop of TD error through critic network
                dEc_dec = e_c
                dec_dV = -1
                dV_dwc_ho = hc
                dV_dwc_ih = wco * 1/2 * (1-hc**2).T * observation[None, :]

                dEc_dwc_ho = dEc_dec * dec_dV * dV_dwc_ho
                dEc_dwc_ih = dEc_dec * dec_dV * dV_dwc_ih

                wci += self.learning_rate * -dEc_dwc_ih.T
                wco += self.learning_rate * -dEc_dwc_ho.T

                # 5. Update actor
                # dEa_dwa = dEa_dea * dea_dst * dst_du * du_dwa   = V(S_t) * dV/ds * ds/da * da/dwa
                #    get new value estimate for current observation with new critic weights
                v, h = _critic(observation)
                #  backpropagate value estimate through critic to input
                dea_dst = wci @ (wco * 1/2 * (1-h**2).T)

                #    get new action estimate with current actor weights
                a, ha = _actor(observation)
                #    backprop action through actor network
                da_dwa_ho = ha.T * (1-a**2)

                da1_dwa1 = (1-a[0]**2) * wao[:, [0]] * (1-ha**2).T * observation[None, :]
                da2_dwa1 = (1-a[1]**2) * wao[:, [1]] * (1-ha**2).T * observation[None, :]

                # old: a_ih = np.deg2rad(10) * 1/2 * (1-a**2) * wao * 1/2 * (1-ha**2).T * observation[None, :]

                #    chain rule to get grad of actor error to actor weights
                dEa_da = v * np.dot(dea_dst.T, dst_da) * self.action_scaling
                dEa_dwa_ho = dEa_da * da_dwa_ho
                dEa_dwa_ih = dEa_da[0, 0] * da1_dwa1 + dEa_da[0, 1] * da2_dwa1

                #    update actor weights
                wai += self.learning_rate * -dEa_dwa_ih.T
                wao += self.learning_rate * -dEa_dwa_ho

            # 6. Statistics
            #if onedof: simple, else this complex one
            stats.append({'t': env.task.t,
                          'x': observation[0],
                          'z': observation[1],
                          'u': observation[2],
                          'w': observation[3],
                          'theta': observation[4],
                          'q': observation[5],
                          'collective': action[0],
                          'cyclic': action[1],
                          'r': reward})

            observation = next_observation
            #  Weights only change slowly, so we can afford not to store 767496743 numbers
            if (step+1) % 10 == 0:
                weight_stats['wci'].append(wci.ravel().copy())
                weight_stats['wco'].append(wco.ravel().copy())
                weight_stats['wai'].append(wai.ravel().copy())
                weight_stats['wao'].append(wao.ravel().copy())

            # Anneal learning rate (optional)
            if anneal_learning_rate and self.learning_rate > 0.01:
                self.learning_rate *= annealing_rate

            if done:
                break
        #  Performance statistics
        episode_stats = pd.DataFrame(stats)
        episode_reward = episode_stats.r.sum()
        weights = {'wci': pd.DataFrame(weight_stats['wci']),
                   'wco': pd.DataFrame(weight_stats['wco']),
                   'wai': pd.DataFrame(weight_stats['wai']),
                   'wao': pd.DataFrame(weight_stats['wao'])}
        #print("Cumulative reward episode:#" + str(self.run_number), episode_reward)
        #  Neural network weights over time, saving only every 10th timestep because the system only evolves slowly
        # if plotstats:
        #     plot_stats(episode_stats, info=info, show_u=True)

        return episode_reward, weights, info, episode_stats

    # def test(self, env, max_steps=None, render=True):
    #     obs, done, ep_reward = env.reset(), False, 0
    #     action=0
    #     stats = []
    #
    #     # TODO: Generalize this to work with any model, not just 1dof heli one
    #     while not done:
    #         stats.append({'t': env.task.t, 'q': obs[0], 'q_ref': env.task.get_q_ref(), 'a1': obs[1], 'u': action})
    #         action = self.action(obs)
    #         obs, reward, done = env.step(action)
    #         ep_reward += reward
    #         max_episode_length = env.max_episode_length if max_steps is None else max_steps
    #
    #     if render:
    #         df = pd.DataFrame(stats)
    #         plot_stats(df, 'test', True)
    #
    #     return ep_reward


if __name__ == '__main__':
    agent = HDPAgentNumpy(n_inputs=6, n_actions=2, weights_std=0.4, learning_rate=0.4)
