import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import numpy as np
import pandas as pd
from functools import partial
import time

from heli_models import Helicopter3DOF
from tasks import HoverTask
from plotting import plot_neural_network_weights_2, plot_stats_3dof, plot_policy_function

tf.keras.backend.set_floatx('float64')


def scaled_tanh(scale, x):
    return tf.tanh(x) * scale


class TFActor(tf.keras.Model):

    def __init__(self, n_hidden, action_scaling, initializer):
        super(TFActor, self).__init__()
        self.h1 = kl.Dense(n_hidden,
                           activation='tanh',
                           use_bias=False,
                           kernel_initializer=initializer)
        self.a = kl.Dense(1,
                          activation=partial(scaled_tanh, action_scaling),
                          use_bias=False,
                          kernel_initializer=initializer)

    def call(self, x):
        x = self.h1(x)
        return self.a(x)


class TFCritic(tf.keras.Model):

    def __init__(self, n_hidden, initializer):
        super(TFCritic, self).__init__()
        self.h1 = kl.Dense(n_hidden, activation='tanh', use_bias=False, kernel_initializer=initializer)
        self.v = kl.Dense(1, name='value', use_bias=False,  kernel_initializer=initializer)

    def call(self, x):
        x = self.h1(x)
        return self.v(x)


class CollectivePID:

    def __init__(self, h_ref=25, dt=0.01, proportional_gain=2, integral_gain=0.2, derivative_gain=0.1):
        self.h_ref = h_ref
        self.hdot_corr = 0
        self.hdot_err = 0
        self.dt = dt
        self.Kp = proportional_gain
        self.Ki = integral_gain
        self.Kd = derivative_gain

    def __call__(self, obs):

        hdot_ref = self.Kd * (self.h_ref - -obs[1])
        hdot = (obs[2] * np.sin(obs[4]) - obs[3] * np.cos(obs[4]))
        self.hdot_err = (hdot_ref - hdot)
        collective = np.deg2rad(5 + self.Kp * self.hdot_err + self.Ki * self.hdot_corr)

        return collective

    def increment_hdot_error(self):
        self.hdot_corr += self.dt * self.hdot_err


class HDPAgentTF:

    def __init__(self,
                 collective_controller,
                 gamma=0.6,
                 lr_actor=0.1,
                 lr_critic=0.1,
                 weights_stddev=0.04,
                 run_number=0,
                 action_scaling=np.deg2rad(15),
                 n_hidden=6,
                 ac_states=(5,)
                 ):
        self.ac_states = ac_states
        self.collective_controller = collective_controller
        initializer = tf.initializers.TruncatedNormal(mean=0.0, stddev=weights_stddev)
        self.actor = TFActor(n_hidden=n_hidden, action_scaling=action_scaling, initializer=initializer)
        self.critic = TFCritic(n_hidden=n_hidden, initializer=initializer)
        self.optimizer_actor = ko.SGD(lr=lr_actor)
        self.optimizer_critic = ko.SGD(lr=lr_critic)
        self.gamma = gamma
        self.action_scaling = action_scaling
        self.info = {'run_number': run_number,
                     'n_hidden': n_hidden,
                     'lr_actor': lr_actor,
                     'lr_critic': lr_critic}
        self.loss_object = kls.MeanSquaredError()
        self.update_networks_flag = True

    def train(self,
              env,
              trim_speed,
              plotstats=True,
              n_updates=5,
              anneal_learning_rate=False,
              annealing_rate=0.9994,
              print_runtime=True):

        t1 = time.time()
        # training loop: collect samples, send to optimizer, repeat updates times
        episode_rewards = [0.0]

        # This is a property of the environment
        ds_da = tf.constant(env.get_environment_transition_function()[(5, 7), 1].reshape(2, 1))

        # Initialize environment and tracking task
        observation, trim_actions = env.reset(v_initial=trim_speed)
        stats = []
        weight_stats = {'t': [],
                        'wci': [],
                        'wco': [],
                        'wai': [],
                        'wao': []}

        # Repeat (for each step t of an episode)
        for step in range(int(env.episode_ticks)):
            if env.t > 50 and self.update_networks_flag is True:
                self.update_networks_flag = False

            # 1. Obtain action from critic network using current knowledge
            q_err = env.task.get_ref() - observation[5]
            #  TODO: make a generic function to augment the state
            s_aug = tf.constant([[observation[self.ac_states], q_err]])

            collective = self.collective_controller(observation)
            cyclic = self.actor(s_aug).numpy().squeeze()

            action = [collective, cyclic]

            # # 3. Perform action, obtain next state and reward info
            next_observation, reward, done = env.step(action)
            next_q_err = env.task.get_ref() - next_observation[5]
            next_aug = tf.constant([[next_observation[self.ac_states], next_q_err]])

            # TD target remains fixed per time-step to avoid oscillations
            td_target = reward + self.gamma * self.critic(next_aug)

            #  Update actor and critic networks after the transition..
            if self.update_networks_flag:
                self.update_networks(td_target, s_aug, ds_da, n_updates)

            #  Logging
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

            #  Next step...
            self.collective_controller.increment_hdot_error()
            observation = next_observation

            #  Weights only change slowly, so we can afford not to store 767496743 numbers
            if (step) % 10 == 0 and self.update_networks_flag:
                weight_stats['t'].append(env.task.t)
                weight_stats['wci'].append(agent.critic.trainable_weights[0].numpy().ravel().copy())
                weight_stats['wco'].append(agent.critic.trainable_weights[1].numpy().ravel().copy())
                weight_stats['wai'].append(agent.actor.trainable_weights[0].numpy().ravel().copy())
                weight_stats['wao'].append(agent.actor.trainable_weights[1].numpy().ravel().copy())

            # # Anneal learning rate (optional)
            # if anneal_learning_rate and self.learning_rate > 0.01:
            #     self.learning_rate *= annealing_rate

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

        if print_runtime:
            print("Episode run time:", time.time() - t1, "seconds")
        return episode_reward, episode_stats, weights

    @tf.function
    def update_networks(self, td_target, s_aug, ds_da, n_updates=2):

        for _ in tf.range(tf.constant(n_updates)):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(s_aug)
                value = self.critic(s_aug)
                value_loss = self.loss_object(td_target, value)
                cyclic = self.actor(s_aug)
            # Critic gradients is the derivative of MSE between td-target and value
            gradient_critic = tape.gradient(value_loss, self.critic.trainable_variables)
            self.optimizer_critic.apply_gradients(zip(gradient_critic, self.critic.trainable_variables))

            # Actor gradient through the critic and environment model: dEa/dwa = V * dV/ds * ds/da * da/dwa
            dV_ds = tape.gradient(value, s_aug)
            da_dwa = tape.gradient(cyclic, self.actor.trainable_variables)
            scale = tf.squeeze(tf.multiply(value, tf.matmul(dV_ds, ds_da)))
            gradient_actor = [tf.multiply(scale, x) for x in da_dwa]
            self.optimizer_actor.apply_gradients(zip(gradient_actor, self.actor.trainable_variables))


if __name__ == "__main__":
    dt = 0.01  # s
    trim_speed = 10  # m/s
    task = HoverTask(dt=dt)
    env = Helicopter3DOF(task=task, t_max=120, dt=dt)
    col = CollectivePID(dt=dt, h_ref=25, derivative_gain=0.2)

    agent = HDPAgentTF(collective_controller=col,
                       gamma=0.6,
                       lr_actor=0.1,
                       lr_critic=0.1)
    reward, episode_stats, weights = agent.train(env, trim_speed=trim_speed, n_updates=1, print_runtime=True)
    plot_stats_3dof(episode_stats, info=agent.info)
    # plot_neural_network_weights_2(weights)
    # q_range = np.deg2rad(np.arange(-5, 5, 0.25))
    # qerr_range = np.deg2rad(np.arange(-2, 2, 0.1))
    #
    # Z = plot_policy_function(agent, q_range, qerr_range)

