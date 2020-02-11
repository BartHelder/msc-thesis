import numpy as np
import pandas as pd
from functools import partial
import time
import datetime
import json
import pickle as pkl
import itertools

from heli_models import Helicopter3DOF
from plotting import plot_neural_network_weights_2, plot_stats_3dof, plot_policy_function
from model import RecursiveLeastSquares

class TFActor3DOF(tf.keras.Model):

    def __init__(self, n_hidden, initializer, action_scaling=15, offset=0):
        super(TFActor3DOF, self).__init__()
        self.action_scaling = np.deg2rad(action_scaling)
        self.offset = np.deg2rad(offset)

        self.h1 = kl.Dense(n_hidden,
                           activation='tanh',
                           use_bias=False,
                           kernel_initializer=initializer)
        self.a = kl.Dense(1,
                          activation=partial(scaled_tanh, self.action_scaling),
                          kernel_initializer=initializer,
                          use_bias=False)

    def call(self, x, **kwargs):
        x = self.h1(x)
        return tf.add(self.offset, self.a(x))


class TFActor6DOF(tf.keras.Model):

    def __init__(self, n_hidden, initializer):
        super(TFActor6DOF, self).__init__()
        self.h1 = kl.Dense(n_hidden,
                           activation='tanh',
                           kernel_initializer=initializer,
                           use_bias=False,
                           bias_initializer=initializer)

        self.a = kl.Dense(1,
                          activation='sigmoid',
                          kernel_initializer=initializer,
                          use_bias=False,
                          )

    def call(self, x, **kwargs):
        x = self.h1(x)
        return self.a(x)


class HDPCritic(tf.keras.Model):

    def __init__(self, n_hidden, initializer):
        super(HDPCritic, self).__init__()
        self.h1 = kl.Dense(n_hidden, activation='tanh', use_bias=False, kernel_initializer=initializer)
        self.v = kl.Dense(1, name='value', use_bias=False,  kernel_initializer=initializer)

    def call(self, x, **kwargs):
        x = self.h1(x)
        return self.v(x)


class DHPCritic(tf.keras.Model):

    def __init__(self, n_hidden, n_outputs, initializer):
        super(DHPCritic, self).__init__()
        self.h1 = kl.Dense(n_hidden, activation='tanh', use_bias=False, kernel_initializer=initializer)
        self.v = kl.Dense(n_outputs, name='value', use_bias=False, kernel_initializer=initializer)

    def call(self, x, **kwargs):
        x = self.h1(x)
        return self.v(x)


class Agent(object):
    def __init__(self, config, control_channel):
        with open(config, "r") as f:
            c = json.load(f)
        conf = c["agent"][control_channel]
        self.optimizer_actor = ko.SGD(lr=conf["lr_actor"],
                                nesterov=bool(conf["use_nesterov_momentum"]),
                                momentum=conf["momentum"])
        self.optimizer_critic = ko.SGD(lr=conf["lr_critic"],
                                nesterov=bool(conf["use_nesterov_momentum"]),
                                momentum=conf["momentum"])
        self.control_channel = control_channel
        self.tracked_state = conf["tracked_state"]
        self.actor_critic_states = conf["actor_critic_states"]
        self.reward_weight = conf["reward_weight"]
        self.discount_factor = conf["discount_factor"]
        self.mapping = {'collective': 0,
                        'cyclic_lon': 1,
                        'cyclic_lat': 2,
                        'directional': 3}
        self.info = {'n_hidden': conf["n_hidden"],
                     'discount_factor': conf["discount_factor"],
                     'lr_actor': conf["lr_actor"],
                     'lr_critic': conf["lr_critic"]}
        self.actor = None
        self.critic = None
        self.model = None

    def augment_state(self, observation, reference):
        tracking_error = reference[self.tracked_state] - observation[self.tracked_state]
        return tf.constant([[observation[x] for x in self.actor_critic_states] + [tracking_error]])

    def get_reward(self, observation, reference, clip_value=-5):
        tracking_error = reference[self.tracked_state] - observation[self.tracked_state]
        reward = -tracking_error**2 * self.reward_weight
        reward = np.clip(reward, clip_value, 0)
        return reward

    def update_networks(self, **kwargs):
        raise NotImplementedError("This function needs to be overwritten by a child class")

    def save(self, name, folder="saved_models/"+datetime.date.today().isoformat()+"/"):
        self.actor.save_weights(folder + "actor_" + name, save_format='tf')
        self.critic.save_weights(folder + "critic_" + name, save_format='tf')

    def load(self, name, folder):
        self.actor.load_weights(folder + "actor_" + name)
        self.critic.load_weights(folder + "critic_" + name)


class HDPAgent(Agent):

    def __init__(self,
                 config,
                 control_channel,
                 actor,
                 **kwargs

                 ):
        super().__init__(config, control_channel)
        with open(config, "r") as f:
            c = json.load(f)
        conf = c["agent"][control_channel]
        initializer = tf.initializers.TruncatedNormal(mean=0.0, stddev=conf["weights_stddev"])

        self.actor = actor(n_hidden=conf["n_hidden"],
                           initializer=initializer,
                           **kwargs)
        self.critic = HDPCritic(n_hidden=conf["n_hidden"],
                                initializer=initializer)
        self.ds_da = None
        self.loss_object = kls.MeanSquaredError()

    def set_ds_da(self, rls_model):


        ds_da = rls_model.gradient_action()[:, self.mapping[self.control_channel]]
        state_transition = ds_da[self.actor_critic_states]
        # if the state increases, tracking error decreases proportionally:
        tracking_error = -ds_da[self.tracked_state].reshape((1, 1))
        dsda = np.append(state_transition, tracking_error, axis=0)
        self.ds_da = tf.constant(dsda)

        return dsda

    def update_networks(self, td_target, s_aug, n_updates=2):
        """
        :param x:
        """
        for _ in tf.range(tf.constant(n_updates)):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(s_aug)
                value = self.critic(s_aug)
                value_loss = self.loss_object(td_target, value)
                action = self.actor(s_aug)

            # Critic gradients is the derivative of MSE between td-target and value
            gradient_critic = tape.gradient(value_loss, self.critic.trainable_variables)
            self.optimizer_critic.apply_gradients(zip(gradient_critic, self.critic.trainable_variables))

            # Actor gradient through the critic and environment model: dEa/dwa = V * dV/ds * ds/da * da/dwa
            dV_ds = tape.gradient(value, s_aug)
            scale = tf.squeeze(tf.multiply(value, tf.matmul(dV_ds, self.ds_da)))
            da_dwa = tape.gradient(action, self.actor.trainable_variables)
            gradient_actor = [tf.multiply(scale, x) for x in da_dwa]
            self.optimizer_actor.apply_gradients(zip(gradient_actor, self.actor.trainable_variables))


class DHPAgent(Agent):

    def __init__(self,
                 config,
                 control_channel,
                 actor,
                 target_critic_tau=0.01,
                 **kwargs
                 ):

        super().__init__(config, control_channel)
        with open(config, "r") as f:
            c = json.load(f)
        conf = c["agent"][control_channel]
        initializer = tf.initializers.TruncatedNormal(mean=0.0, stddev=conf["weights_stddev"])
        self.actor = actor(n_hidden=conf["n_hidden"],
                           initializer=initializer,
                           **kwargs)
        self.critic = DHPCritic(n_hidden=conf["n_hidden"],
                                n_outputs=len(conf["actor_critic_states"])+1,
                                initializer=initializer)
        # TODO: copy critic weights to target critic --after-- initialization
        self.target_critic = DHPCritic(n_hidden=conf["n_hidden"],
                             n_outputs=len(conf["actor_critic_states"])+1,
                             initializer=initializer)
        self.critic_tau = target_critic_tau

    def get_reward(self, observation, reference, clip_value=-5):
        tracking_error = reference[self.tracked_state] - observation[self.tracked_state]
        reward = -tracking_error**2 * self.reward_weight
        reward = np.clip(reward, clip_value, 0)
        reward_derivative = np.zeros(shape=(1, len(self.actor_critic_states)+1))
        reward_derivative[-1] = -2 * tracking_error * self.reward_weight
        return reward, reward_derivative

    def update_networks(self, s1, s2, F, G, dr_ds):

        # Forward passes
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(s1)
            lambda_1 = self.critic(s1)
            action = self.actor(s1)

        lambda_2 = self.target_critic(s2)

        # Backward pass
        dlambda_dwc = tape.gradient(lambda_1, self.critic.trainable_variables)
        da_ds = tape.gradient(action, s1)
        da_dwa = tape.gradient(action, self.actor.trainable_variables)
        del tape

        # Select relevant rows and columns from F and G matrices
        state_indices = self.actor_critic_states + [self.tracked_state]
        Fhat = F[np.ix_(state_indices, state_indices)]
        Ghat = G[np.ix_(state_indices, [self.mapping[self.control_channel]])]

        # Critic update
        target = dr_ds + self.discount_factor * lambda_2
        e_c = lambda_1 - dr_ds - target @ (Fhat + Ghat @ da_ds)
        gradients_critic = e_c @ dlambda_dwc
        self.optimizer_critic.apply_gradients(zip(gradients_critic, self.critic.trainable_variables))

        # Sync target network
        for target_weights, weights in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            target_weights = self.critic_tau * weights + (1-self.critic_tau) * target_weights

        # Actor update
        gradients_actor = -target * G * da_dwa
        self.optimizer_actor.apply_gradients(zip(gradients_actor, self.actor.trainable_variables))


        return
