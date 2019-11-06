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
                 weights_std=0.1,
                 n_hidden=6,
                 n_inputs_critic=6,
                 inputs_actor1=(1, 2, 3, 4),
                 inputs_actor2=(0, 2, 4, 5)):
        self.gamma = discount_factor
        self.learning_rate = learning_rate
        self.action_scaling = action_scaling

        self.inputs_actor1 = inputs_actor1
        self.inputs_actor2 = inputs_actor2

        self.w_critic_input_to_hidden = np.random.randn(n_inputs_critic, n_hidden) * weights_std
        self.w_critic_hidden_to_output = np.random.randn(n_hidden, 1) * weights_std

        self.w_actor1_input_to_hidden = np.random.randn(len(inputs_actor1), n_hidden) * weights_std
        self.w_actor1_hidden_to_output = np.random.randn(n_hidden, 1) * weights_std
        self.w_actor2_input_to_hidden = np.random.randn(len(inputs_actor2), n_hidden) * weights_std
        self.w_actor2_hidden_to_output = np.random.randn(n_hidden, 1) * weights_std

        self.run_number = run_number

    def action(self, obs):

        z1 = np.matmul(obs[None, :], self.w_actor1_input_to_hidden)
        h = np.tanh(z1)
        a2 = np.matmul(h, self.w_actor1_hidden_to_output)

        return (self.action_scaling * np.tanh(a2)).squeeze()

    def train(self, env, trim_speed, plotstats=True, n_updates=5, anneal_learning_rate=False, annealing_rate=0.9994):

        # training loop: collect samples, send to optimizer, repeat updates times
        episode_rewards = [0.0]

        wci = self.w_critic_input_to_hidden
        wco = self.w_critic_hidden_to_output
        wa1i = self.w_actor1_input_to_hidden
        wa1o = self.w_actor1_hidden_to_output
        wa2i = self.w_actor2_input_to_hidden
        wa2o = self.w_actor2_hidden_to_output
        wa = [wa1i, wa1o, wa2i, wa2o]

        # This is a property of the environment
        dst_da = env.get_environment_transition_function()

        def _critic(obs):
            c_hidden_in = np.matmul(obs[None, :], wci)
            hidden = np.tanh(c_hidden_in)
            value = np.matmul(hidden, wco)

            return value.squeeze(), hidden

        def _actor1(obs, scale=self.action_scaling):
            #  Actor 1: collective control, regulates altitude.
            #  u_collective = f(u, w, pitch, z_err)
            a1 = np.matmul(obs[None, self.inputs_actor1], wa1i)
            h = np.tanh(a1)
            a2 = np.matmul(h, wa1o)

            return (scale * np.tanh(a2)).squeeze(), h

        def _actor2(obs, scale=self.action_scaling):
            #  Actor 2: cyclic control for attitude, velocity, and ground location
            # u_cyclic = f(u_err, u, x_err, theta_err, q)
            z1 = np.matmul(obs[None, self.inputs_actor2], wa2i)
            h = np.tanh(z1)
            z2 = np.matmul(h, wa2o)

            return (scale * np.tanh(z2)).squeeze(), h

        # Initialize environment and tracking task
        observation, trim_actions = env.reset(v_initial=trim_speed)
        stats = []
        weight_stats = {'t': [0],
                        'wci': [wci.ravel().copy()],
                        'wco': [wco.ravel().copy()],
                        'wa1i': [wa1i.ravel().copy()],
                        'wa1o': [wa1o.ravel().copy()],
                        'wa2i': [wa2i.ravel().copy()],
                        'wa2o': [wa2o.ravel().copy()]}

        info = {'run_number': self.run_number,
                'learning_rate': self.learning_rate,
                'stats': env.stats}

        # Repeat (for each step t of an episode)
        for step in range(int(env.episode_ticks)):

            #  Weights only change slowly, so we can afford not to store 767496743 numbers
            if (step+1) % 10 == 0:
                weight_stats['t'].append(env.task.t)
                weight_stats['wci'].append(wci.ravel().copy())
                weight_stats['wco'].append(wco.ravel().copy())
                weight_stats['wa1i'].append(wa1i.ravel().copy())
                weight_stats['wa1o'].append(wa1o.ravel().copy())
                weight_stats['wa2i'].append(wa2i.ravel().copy())
                weight_stats['wa2o'].append(wa2o.ravel().copy())

            # 1. Obtain action from actor network using current knowledge
            collective, hidden_col = _actor1(observation, scale=self.action_scaling)
            cyclic, hidden_cyc = _actor2(observation, scale=self.action_scaling)
            # action = [collective, cyclic]
            action = [0, cyclic]

            # 2. Obtain value estimate for current state
            # value, hidden_v = _critic(observation)

            # 3. Perform action, obtain next state and reward info
            next_observation, reward, done = env.step(trim_actions + action)

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
                dV_dwc_ih = wco * (1-hc**2).T * observation[None, :]

                dEc_dwc_ho = dEc_dec * dec_dV * dV_dwc_ho
                dEc_dwc_ih = dEc_dec * dec_dV * dV_dwc_ih

                wci += self.learning_rate * -dEc_dwc_ih.T
                wco += self.learning_rate * -dEc_dwc_ho.T

                # 5. Update actor

                # old: a_ih = np.deg2rad(10) * 1/2 * (1-a**2) * wao * 1/2 * (1-ha**2).T * observation[None, :]
                if step % 100 == 0:
                    dst_da = env.get_environment_transition_function()

                # dEa_dwa = dEa_dea * dea_dst * dst_du * du_dwa   = V(S_t) * dV/ds * ds/da * da/dwa
                #    get new value estimate for current observation with new critic weights
                v, h = _critic(observation)
                #  backpropagate value estimate through critic to input
                dea_dst = wci @ (wco * (1-h**2).T)

                #    get new action estimate with current actor weights
                collective, hidden_col = _actor1(observation, scale=self.action_scaling)
                cyclic, hidden_cyc = _actor2(observation, scale=self.action_scaling)

                #    backprop action through actor networks
                da1_dwa2 = hidden_col.T * (1-collective**2)
                da1_dwa1 = (1-collective**2) * wa1o * (1-hidden_col**2).T * observation[None, self.inputs_actor1]
                da2_dwa2 = hidden_cyc.T * (1-cyclic**2)
                da2_dwa1 = (1-cyclic**2) * wa2o * (1-hidden_cyc**2).T * observation[None, self.inputs_actor2]

                dEa_da = (v * np.dot(dea_dst.T, dst_da) * self.action_scaling).ravel()
                grads = [da1_dwa1.T*dEa_da[0],
                         da1_dwa2*dEa_da[0],
                         da2_dwa1.T*dEa_da[1],
                         da2_dwa2*dEa_da[1]]

                #    update actor weights
                wa = [x + self.learning_rate*y for x, y in zip(wa, grads)]

            # 6. Statistics
            #if onedof: simple, else this complex one
            stats.append({'t': env.t,
                          'x': observation[0],
                          'z': observation[1],
                          'u': observation[2],
                          'w': observation[3],
                          'theta': observation[4],
                          'q': observation[5],
                          'collective': action[0] + trim_actions[0],
                          'cyclic': action[1] + trim_actions[1],
                          'r': reward})

            observation = next_observation
            # Anneal learning rate (optional)
            if anneal_learning_rate and self.learning_rate > 0.01:
                self.learning_rate *= annealing_rate

            if done:
                break
        #  Performance statistics
        episode_stats = pd.DataFrame(stats)
        episode_reward = episode_stats.r.sum()
        weights = {'wci': pd.DataFrame(data=weight_stats['wci'], index=weight_stats['t']),
                   'wco': pd.DataFrame(data=weight_stats['wco'], index=weight_stats['t']),
                   'wa1i': pd.DataFrame(data=weight_stats['wa1i'], index=weight_stats['t']),
                   'wa1o': pd.DataFrame(data=weight_stats['wa1o'], index=weight_stats['t']),
                   'wa2i': pd.DataFrame(data=weight_stats['wa2i'], index=weight_stats['t']),
                   'wa2o': pd.DataFrame(data=weight_stats['wa2o'], index=weight_stats['t'])}

        #print("Cumulative reward episode:#" + str(self.run_number), episode_reward)
        #  Neural network weights over time, saving only every 10th timestep because the system only evolves slowly
        # if plotstats:
        #     plot_stats(episode_stats, info=info, show_u=True)

        return episode_reward, weights, info, episode_stats


class HDPAgentNumpySplit:

    def __init__(self,
                 discount_factor=0.95,
                 learning_rate=0.1,
                 run_number=0,
                 action_scaling=np.deg2rad(15),
                 weights_std=0.1,
                 n_hidden=6,
                 n_inputs_critic=6,
                 inputs_actor=(5,)):  # x, z, u, w, pitch, q, lambdai
        self.gamma = discount_factor
        self.learning_rate = learning_rate
        self.action_scaling = action_scaling

        self.inputs_actor = inputs_actor

        self.w_critic_input_to_hidden = np.random.randn(len(inputs_actor)+1, n_hidden) * weights_std
        self.w_critic_hidden_to_output = np.random.randn(n_hidden, 1) * weights_std

        self.w_actor_input_to_hidden = np.random.randn(len(inputs_actor)+1, n_hidden) * weights_std
        self.w_actor_hidden_to_output = np.random.randn(n_hidden, 1) * weights_std

        self.run_number = run_number

    def action(self, obs):

        z1 = np.matmul(obs[None, :], self.w_actor_input_to_hidden)
        h = np.tanh(z1)
        z2 = np.matmul(h, self.w_actor_hidden_to_output)

        return (self.action_scaling * np.tanh(z2)).squeeze()

    def train(self, env, trim_speed, plotstats=True, n_updates=5, anneal_learning_rate=False, annealing_rate=0.9994):

        # training loop: collect samples, send to optimizer, repeat updates times
        episode_rewards = [0.0]

        wci = self.w_critic_input_to_hidden
        wco = self.w_critic_hidden_to_output
        wai = self.w_actor_input_to_hidden
        wao = self.w_actor_hidden_to_output

        # This is a property of the environment
        dst_da = env.get_environment_transition_function()

        def _critic(obs):
            z1 = np.matmul(obs, wci)
            hidden = np.tanh(z1)
            value = np.matmul(hidden, wco)

            return value.squeeze(), hidden

        def _actor(obs, scale=self.action_scaling):
            #  Actor 2: cyclic control
            #  u_collective = f(u, w, pitch, z_err)
            a1 = np.matmul(obs, wai)
            h = np.tanh(a1)
            a2 = np.matmul(h, wao)

            return (scale * np.tanh(a2)).squeeze(), h

        # Initialize environment and tracking task
        observation, trim_actions = env.reset(v_initial=trim_speed)
        hdot_corr = 0
        hdot_ref = 0
        hdot = 0
        stats = []
        weight_stats = {'t': [0],
                        'wci': [wci.ravel().copy()],
                        'wco': [wco.ravel().copy()],
                        'wai': [wai.ravel().copy()],
                        'wao': [wao.ravel().copy()]}

        info = {'run_number': self.run_number,
                'learning_rate': self.learning_rate,
                'stats': env.stats}

        # Repeat (for each step t of an episode)
        for step in range(int(env.episode_ticks)):

            # if env.t > 90:
            #     self.learning_rate = 0

            #  Weights only change slowly, so we can afford not to store 767496743 numbers
            if (step+1) % 10 == 0:
                weight_stats['t'].append(env.task.t)
                weight_stats['wci'].append(wci.ravel().copy())
                weight_stats['wco'].append(wco.ravel().copy())
                weight_stats['wai'].append(wai.ravel().copy())
                weight_stats['wao'].append(wao.ravel().copy())

            # 1. Obtain action from actor network using current knowledge
            q_err = env.task.get_ref() - observation[5]
            s_aug = np.append(observation[None, [self.inputs_actor]], q_err)[None, :]

            cyclic, hidden_cyc = _actor(s_aug, scale=self.action_scaling)
            # action = [collective, cyclic]
            h_ref = 25

            h = -observation[1]
            hdot_ref = 0.1 * (h_ref - h)
            hdot = (observation[2] * np.sin(observation[4]) - observation[3] * np.cos(observation[4]))
            collective = np.deg2rad(trim_actions[0] + 2 * (hdot_ref - hdot) + 0.3 * hdot_corr)

            action = [collective, cyclic]

            # 2. Obtain value estimate for current state
            # value, hidden_v = _critic(observation)

            # 3. Perform action, obtain next state and reward info
            next_observation, reward, done = env.step(action)
            next_q_err = env.task.get_ref() - next_observation[5]
            next_aug = np.append(next_observation[None, [self.inputs_actor]], next_q_err)[None, :]

            # TD target remains fixed per time-step to avoid oscillations
            td_target = reward + self.gamma * _critic(next_aug)[0]


            # Update models x times per timestep
            for j in range(n_updates):

                # 4. Update critic: error is td_target minus value estimate for current observation
                v, hc = _critic(s_aug)
                e_c = td_target - v

                #  dEc/dwc = dEc/dec * dec/dV * dV/dwc  = standard backprop of TD error through critic network
                dEc_dec = e_c
                dec_dV = -1
                dV_dwc_ho = hc
                dV_dwc_ih = wco * (1-hc**2).T * s_aug

                dEc_dwc_ho = dEc_dec * dec_dV * dV_dwc_ho
                dEc_dwc_ih = dEc_dec * dec_dV * dV_dwc_ih

                wci += self.learning_rate * -dEc_dwc_ih.T
                wco += self.learning_rate * -dEc_dwc_ho.T

                # 5. Update actor
                # old: a_ih = np.deg2rad(10) * 1/2 * (1-a**2) * wao * 1/2 * (1-ha**2).T * observation[None, :]
                if step % 100 == 0:
                    dst_da = env.get_environment_transition_function()[(self.inputs_actor + (-1,)), 1]

                # dEa_dwa = dEa_dea * dea_dst * dst_du * du_dwa   = V(S_t) * dV/ds * ds/da * da/dwa
                #    get new value estimate for current observation with new critic weights
                v, hc = _critic(s_aug)
                #  backpropagate value estimate through critic to input
                dea_dst = wci @ (wco * (1-hc**2).T)

                #    get new action estimate with current actor weights
                cyclic, hidden_cyc = _actor(s_aug, scale=self.action_scaling)

                #    backprop action through actor networks
                da_dwa2 = hidden_cyc.T * (1-cyclic**2)
                da_dwa1 = (1-cyclic**2) * wao * (1-hidden_cyc**2).T * s_aug

                dEa_da = (v * np.dot(dea_dst.T, dst_da) * self.action_scaling).ravel()

                #    update actor weights
                wao += self.learning_rate * -(dEa_da * da_dwa2)
                wai += self.learning_rate * -(dEa_da * da_dwa1.T)

            # 6. Statistics
            #if onedof: simple, else this complex one
            stats.append({'t': env.t,
                          'x': observation[0],
                          'z': observation[1],
                          'u': observation[2],
                          'w': observation[3],
                          'theta': observation[4],
                          'q': observation[5],
                          'qref': env.task.get_ref(),
                          'collective': action[0],
                          'cyclic': action[1],
                          'r': reward})

            hdot_corr += env.dt * (hdot_ref - hdot)
            observation = next_observation
            # Anneal learning rate (optional)
            if anneal_learning_rate and self.learning_rate > 0.01:
                self.learning_rate *= annealing_rate

            if done:
                break
        #  Performance statistics
        episode_stats = pd.DataFrame(stats)
        episode_reward = episode_stats.r.sum()
        weights = {'wci': pd.DataFrame(data=weight_stats['wci'], index=weight_stats['t']),
                   'wco': pd.DataFrame(data=weight_stats['wco'], index=weight_stats['t']),
                   'wai': pd.DataFrame(data=weight_stats['wai'], index=weight_stats['t']),
                   'wao': pd.DataFrame(data=weight_stats['wao'], index=weight_stats['t'])}

        #print("Cumulative reward episode:#" + str(self.run_number), episode_reward)
        #  Neural network weights over time, saving only every 10th timestep because the system only evolves slowly
        # if plotstats:
        #     plot_stats(episode_stats, info=info, show_u=True)

        return episode_reward, weights, info, episode_stats


if __name__ == '__main__':
    agent = HDPAgentNumpy(n_actions=2, weights_std=0.4, learning_rate=0.4, n_inputs_actor1=4, n_inputs_actor2=4)
