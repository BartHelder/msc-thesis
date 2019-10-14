import numpy as np
from plotting import plot_stats, plot_neural_network_weights
import pandas as pd
from collections import deque
from heli_models import Helicopter1DOF


class HDPAgentNumpy:

    def __init__(self,
                 discount_factor=0.95,
                 learning_rate=0.1,
                 run_number=0,
                 action_scaling=np.deg2rad(10),
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

        def _actor(obs, scale=1.0):
            a_hidden_in = np.matmul(obs[None, :], wai)
            hidden = np.tanh(a_hidden_in)
            a_out_in = np.matmul(hidden, wao)
            act = scale * np.tanh(a_out_in)

            return act.squeeze(), hidden

        # Initialize environment and tracking task
        observation = env.reset()
        stats = []
        weight_stats = []
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
            next_observation, reward, done = env.step(action)

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
                da_dwa_ho = np.deg2rad(10) * 1/2 * (1-a**2) * ha
                da_dwa_ih = np.deg2rad(10) * 1/2 * (1-a**2) * wao * 1/2 * (1-ha**2).T * observation[None, :]

                #    chain rule to get grad of actor error to actor weights
                scale = v * (dst_da @ dea_dst)
                dEa_dwa_ho = scale * da_dwa_ho.T
                dEa_dwa_ih = scale * da_dwa_ih.T

                #    update actor weights
                wai += self.learning_rate * -dEa_dwa_ih
                wao += self.learning_rate * -dEa_dwa_ho

            # TODO: Generalize this to work with any model, not just 1dof heli one
            # 6. Statistics
            stats.append({'t': env.task.t,
                          'state': observation,
                          'action': action,
                          'r': reward,
                          'next_state': next_observation,
                          'goal_state': env.task.get_ref()
                          })

            observation = next_observation
            #  Weights only change slowly, so we can afford not to store 6000x49 numbers
            if (step+1) % 10 == 0:
                weight_stats.append(np.concatenate([i.ravel() for i in [wci, wco, wai, wao]]))

            # Anneal learning rate (optional)
            if anneal_learning_rate and self.learning_rate > 0.01:
                self.learning_rate *= annealing_rate

        #  Performance statistics
        episode_stats = pd.DataFrame(stats)
        episode_reward = episode_stats.r.sum()
        #print("Cumulative reward episode:#" + str(self.run_number), episode_reward)
        #  Neural network weights over time, saving only every 10th timestep because the system only evolves slowly
        weights_history = pd.DataFrame(data=weight_stats,
                                       index=np.arange(0, env.max_episode_length, env.dt*10))
        if plotstats:
            plot_stats(episode_stats, info=info, show_u=True)

        return episode_reward, weights_history, info

    def test(self, env, max_steps=None, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        action=0
        stats = []

        # TODO: Generalize this to work with any model, not just 1dof heli one
        while not done:
            stats.append({'t': env.task.t, 'q': obs[0], 'q_ref': env.task.get_q_ref(), 'a1': obs[1], 'u': action})
            action = self.action(obs)
            obs, reward, done = env.step(action)
            ep_reward += reward
            max_episode_length = env.max_episode_length if max_steps is None else max_steps

        if render:
            df = pd.DataFrame(stats)
            plot_stats(df, 'test', True)

        return ep_reward


if __name__ == '__main__':
    agent = HDPAgentNumpy(n_inputs=6, n_actions=2, weights_std=0.4, learning_rate=0.4)
