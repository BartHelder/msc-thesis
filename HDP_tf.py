from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import pandas as pd
from functools import partial


def scaled_tanh(scale, x):
    return tf.keras.backend.tanh(x) * scale


class TFActor(tf.keras.Model):

    def __init__(self, n_hidden, action_scaling):
        super(TFActor, self).__init__()
        self.h1 = kl.Dense(n_hidden, activation='tanh')
        self.a = kl.Dense(2, activation=partial(scaled_tanh, action_scaling))

    def call(self, x):
        x = self.h1(x)
        return self.a(x)


class TFCritic(tf.keras.Model):

    def __init__(self, n_hidden):
        super(TFCritic, self).__init__()
        self.h1 = kl.Dense(n_hidden, activation='tanh')
        self.v = kl.Dense(1, name='value')

    def call(self, x):
        x = self.h1(x)
        return self.v(x)


class HDPAgentTF:

    def __init__(self,
                 discount_factor=0.95,
                 learning_rate_actor=0.1,
                 learning_rate_critic=0.1,
                 run_number=0,
                 action_scaling=np.deg2rad(15),
                 n_hidden=6):
        self.actor = TFActor(n_hidden=n_hidden, action_scaling=action_scaling)
        self.critic = TFCritic(n_hidden=n_hidden)
        self.optimizer_actor = ko.Adam(lr=learning_rate_actor)
        self.optimizer_critic = ko.Adam(lr=learning_rate_critic)
        self.gamma = discount_factor
        self.action_scaling = action_scaling
        self.info = {'run_number': run_number,
                     'n_hidden': n_hidden,
                     'lr_actor': learning_rate_actor,
                     'lr_critic': learning_rate_critic}

    @tf.function
    def train(self,
              env,
              trim_speed,
              plotstats=True,
              n_updates=5,
              anneal_learning_rate=False,
              annealing_rate=0.9994):

        # training loop: collect samples, send to optimizer, repeat updates times
        episode_rewards = [0.0]

        # This is a property of the environment
        dst_da = env.get_environment_transition_function()

        # Initialize environment and tracking task
        observation, trim_actions = env.reset(v_initial=trim_speed)
        stats = []

        # Repeat (for each step t of an episode)
        for step in range(int(env.episode_ticks)):

            # 1. Obtain action from critic network using current knowledge
            with tf.GradientTape() as tape:
                action = self.actor()
            action, hidden_action = _actor(observation, scale=self.action_scaling)

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
                # dEa_dwa = dEa_dea * dea_dst * dst_du * du_dwa   = V(S_t) * dV/ds * ds/da * da/dwa
                #    get new value estimate for current observation with new critic weights
                v, h = _critic(observation)
                #  backpropagate value estimate through critic to input
                dea_dst = wci @ (wco * (1-h**2).T)

                #    get new action estimate with current actor weights
                a, ha = _actor(observation)
                #    backprop action through actor network
                da_dwa_ho = ha.T * (1-a**2)

                daN_dwa1 = [0] * len(a)
                for j in range(len(a)):
                    daN_dwa1[j] = (1-a[j]**2) * wao[:, [j]] * (1-ha**2).T * observation[None, :]

                # old: a_ih = np.deg2rad(10) * 1/2 * (1-a**2) * wao * 1/2 * (1-ha**2).T * observation[None, :]
                if step % 100 == 0:
                    dst_da = env.get_environment_transition_function()
                #    chain rule to get grad of actor error to actor weights
                dEa_da = v * np.dot(dea_dst.T, dst_da) * self.action_scaling
                dEa_dwa_ho = dEa_da * da_dwa_ho
                dEa_dwa_ih = sum([-x*y for x, y in zip(dEa_da.ravel(), daN_dwa1)])

                #    update actor weights
                wai += self.learning_rate * dEa_dwa_ih.T
                wao += self.learning_rate * dEa_dwa_ho

            # 6. Statistics
            #if onedof: simple, else this complex one
            stats.append({'t': env.task.t,
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
            #  Weights only change slowly, so we can afford not to store 767496743 numbers
            if (step+1) % 10 == 0:
                weight_stats['t'].append(env.task.t)
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
        weights = {'wci': pd.DataFrame(data=weight_stats['wci'], index=weight_stats['t']),
                   'wco': pd.DataFrame(data=weight_stats['wco'], index=weight_stats['t']),
                   'wai': pd.DataFrame(data=weight_stats['wai'], index=weight_stats['t']),
                   'wao': pd.DataFrame(data=weight_stats['wao'], index=weight_stats['t'])}
        #print("Cumulative reward episode:#" + str(self.run_number), episode_reward)
        #  Neural network weights over time, saving only every 10th timestep because the system only evolves slowly
        # if plotstats:
        #     plot_stats(episode_stats, info=info, show_u=True)

        return episode_reward, weights, info, episode_stats


if __name__ == "__main__":
    actor = TFActor(n_hidden=6, action_scaling=1)
    loss_object = kls.MeanSquaredError()
    state = np.array([1, 2, 3, 4, 5])
    with tf.GradientTape() as tape:
        a = actor(state[None, :])

    gradients = tape.gradient(a, actor.trainable_variables)

