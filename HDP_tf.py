import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import numpy as np
import pandas as pd
from functools import partial
import time
import datetime
import json
import pickle as pkl

from heli_models import Helicopter3DOF
from plotting import plot_neural_network_weights_2, plot_stats_3dof, plot_policy_function

tf.keras.backend.set_floatx('float64')

COLLECTIVE_STATES = (3,)
COLLECTIVE_TRACKED_STATE = (1,)

CYCLIC_STATES = (4, 5)
CYCLIC_TRACKED_STATE = (4,)



def scaled_tanh(scale, x):
    return tf.tanh(x) * scale


class TFActorCyclic(tf.keras.Model):

    def __init__(self, n_hidden, action_scaling, initializer):
        super(TFActorCyclic, self).__init__()
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


class TFActorColl(tf.keras.Model):

    def __init__(self, n_hidden, initializer):
        super(TFActorColl, self).__init__()
        self.h1 = kl.Dense(n_hidden,
                           activation='tanh',
                           use_bias=False,
                           kernel_initializer=initializer)
        self.a = kl.Dense(1,
                          activation=partial(scaled_tanh, np.deg2rad(5)),
                          kernel_initializer=initializer,
                          use_bias=False)

    def call(self, x):
        x = self.h1(x)
        return np.deg2rad(5) + self.a(x)


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
                 config_path,
                 run_number=0
                 ):

        with open(config_path, "r") as f:
            c = json.load(f)
        conf = c["agent"]

        self.collective_controller = collective_controller
        self.lr_actor = conf["lr_actor"]
        self.lr_critic = conf["lr_critic"]
        initializer = tf.initializers.TruncatedNormal(mean=0.0, stddev=conf["weights_stddev"])
        self.actor_cyclic = TFActorCyclic(n_hidden=conf["n_hidden"],
                                          action_scaling=np.deg2rad(conf["action_scaling"]),
                                          initializer=initializer)
        self.actor_collective = TFActorColl(n_hidden=conf["n_hidden"],
                                            initializer=initializer)
        self.critic = TFCritic(n_hidden=conf["n_hidden"],
                               initializer=initializer)
        self.optimizer_actor_cyclic = ko.SGD(lr=self.lr_actor,
                                             nesterov=bool(conf["use_nesterov_momentum"]),
                                             momentum=conf["momentum"])
        self.optimizer_actor_collective = ko.SGD(lr=self.lr_actor,
                                                 nesterov=bool(conf["use_nesterov_momentum"]),
                                                 momentum=conf["momentum"])
        self.optimizer_critic = ko.SGD(lr=self.lr_critic,
                                       nesterov=bool(conf["use_nesterov_momentum"]),
                                       momentum=conf["momentum"])
        self.gamma = conf["gamma"]
        self.info = {'run_number': run_number,
                     'n_hidden': conf["n_hidden"],
                     'gamma': conf["gamma"],
                     'lr_actor': conf["lr_actor"],
                     'lr_critic': conf["lr_critic"]}
        self.loss_object = kls.MeanSquaredError()
        self.update_networks_flag = True

    def train(self,
              env,
              trim_speed,
              stop_training_time,
              n_updates=1,
              print_runtime=True,
              ):

        t1 = time.time()

        # This is a property of the environment
        ds_da = tf.constant(env.get_environment_transition_function()[(4, 5, 7), 1].reshape(3, 1))
        ds_da2 = tf.constant(np.array([[-0.8], [1.0]]))
        #actor_critic_states = env.task.selected_states

        # Initialize environment and tracking task
        observation, trim_actions = env.reset(v_initial=trim_speed)

        def augment_state(obs, control_channel):

            if control_channel == 'cyclic':
                actor_states = CYCLIC_STATES
                tracked_state = CYCLIC_TRACKED_STATE
                tracking_error = env.get_ref()[tracked_state] - obs[tracked_state]
            elif control_channel == 'collective':
                actor_states = COLLECTIVE_STATES
                tracked_state = COLLECTIVE_TRACKED_STATE
                tracking_error = env.get_ref()[tracked_state] - obs[tracked_state]
            else:
                raise NameError("Invalid control channel name")


            augmented_state = [[obs[x] for x in actor_states] + [tracking_error]]

            return tf.constant(augmented_state)

        #  Track statistics
        stats = []
        weight_stats = {'t': [], 'wci': [], 'wco': [], 'wai': [], 'wao': []}

        # Repeat (for each step t of an episode)
        for step in range(int(env.episode_ticks)):
            # if env.t > stop_training_time and self.update_networks_flag is True:
            #     self.update_networks_flag = False

            # 1. Obtain action from critic network using current knowledge
            s_aug = augment_state(observation, control_channel="cyclic")
            s_aug_collective = augment_state(observation, control_channel='collective')

            collective = self.actor_collective(s_aug_collective).numpy().squeeze()
            cyclic = self.actor_cyclic(s_aug).numpy().squeeze()

            action = [collective, cyclic]

            # # 3. Perform action, obtain next state and reward info
            next_observation, reward, done = env.step(action)
            next_aug = augment_state(next_observation, control_channel="collective")

            # TD target remains fixed per time-step to avoid oscillations
            td_target = reward + self.gamma * self.critic(next_aug)

            #  Update actor and critic networks after the transition..
            if self.update_networks_flag:
                self.update_networks(td_target, s_aug, s_aug_collective, ds_da2, n_updates)

            #  Logging
            stats.append({'t': env.t,
                          'x': observation[0],
                          'z': observation[1],
                          'u': observation[2],
                          'w': observation[3],
                          'theta': observation[4],
                          'q': observation[5],
                          'reference': env.get_ref(),
                          'collective': action[0],
                          'cyclic': action[1],
                          'r': reward})

            #  Next step...
            self.collective_controller.increment_hdot_error()
            observation = next_observation

            if 100 < env.t < 100.04:
                print(observation)
            #  Weights only change slowly, so we can afford not to store 767496743 numbers
            if self.update_networks_flag and (step % 10 == 0):
                weight_stats['t'].append(env.t)
                weight_stats['wci'].append(self.critic.trainable_weights[0].numpy().ravel().copy())
                weight_stats['wco'].append(self.critic.trainable_weights[1].numpy().ravel().copy())
                weight_stats['wai'].append(self.actor_collective.trainable_weights[0].numpy().ravel().copy())
                weight_stats['wao'].append(self.actor_collective.trainable_weights[1].numpy().ravel().copy())

            if done or env.t > stop_training_time:
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
    def update_networks(self, td_target, s_aug1, s_aug2, ds_da, n_updates=2):

        for _ in tf.range(tf.constant(n_updates)):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch([s_aug1, s_aug2])
                value = self.critic(s_aug2)
                value_loss = self.loss_object(td_target, value)
                #cyclic = self.actor_cyclic(s_aug1)
                collective = self.actor_collective(s_aug2)

            # Critic gradients is the derivative of MSE between td-target and value
            gradient_critic = tape.gradient(value_loss, self.critic.trainable_variables)
            self.optimizer_critic.apply_gradients(zip(gradient_critic, self.critic.trainable_variables))

            # Actor gradient through the critic and environment model: dEa/dwa = V * dV/ds * ds/da * da/dwa
            dV_ds = tape.gradient(value, s_aug2)
            scale = tf.squeeze(tf.multiply(value, tf.matmul(dV_ds, ds_da)))

            #da1_dwa = tape.gradient(cyclic, self.actor_cyclic.trainable_variables)
            da2_dwa = tape.gradient(collective, self.actor_collective.trainable_variables)
            #gradient_actor1 = [tf.multiply(scale, x) for x in da1_dwa]
            gradient_actor2 = [tf.multiply(scale, x) for x in da2_dwa]

            #self.optimizer_actor_cyclic.apply_gradients(zip(gradient_actor1, self.actor_cyclic.trainable_variables))
            self.optimizer_actor_collective.apply_gradients(zip(gradient_actor2, self.actor_collective.trainable_variables))

    def save(self, path="saved_models/"+datetime.date.today().isoformat()+"/"):
        self.actor_cyclic.save_weights(path + "actor", save_format='tf')
        self.critic.save_weights(path + "critic", save_format='tf')

    def load(self, path):
        self.actor_cyclic.load_weights(path + "actor")
        #self.critic.load_weights(path + "critic")


def train_save_pitch(seed, save_path, config_path="config.json"):

    with open(config_path, 'r') as f:
        config = json.load(f)

    dt = config["dt"]  # s
    tracked_states = [4, 5]
    state_weights = [100, 0]
    tf.random.set_seed(seed)

    col = CollectivePID(dt=dt, h_ref=25, derivative_gain=0.2)
    agent = HDPAgentTF(collective_controller=col, config_path=config_path)
    env = Helicopter3DOF(t_max=120, dt=dt)
    env.task['type'] = 'sinusoid'
    reward, episode_stats, weights = agent.train(env, stop_training_time=120, trim_speed=10, n_updates=1, print_runtime=True)
    if save_path is not None:
        agent.save(save_path)
    else:
        agent.save()

    # Plotting
    plot_stats_3dof(episode_stats, info=agent.info, results_only=True)
    plot_neural_network_weights_2(weights)


if __name__ == "__main__":

    dt = 0.02  # s
    #tf.random.set_seed()

    cfp = "config.json"

    env = Helicopter3DOF(t_max=120, dt=dt)
    env.setup_from_config(task="sinusoid", config_path=cfp)
    col = CollectivePID(dt=dt, h_ref=0, derivative_gain=0.3)
    agent = HDPAgentTF(collective_controller=col, config_path=cfp)

    agent.load("saved_models/2019-11-27/")
    reward2, episode_stats2, weights2 = agent.train(env, stop_training_time=120, trim_speed=1, n_updates=1, print_runtime=True)
    plot_stats_3dof(episode_stats2, info=agent.info, results_only=False)


    # for episode in range(20, 41, 1):
    #     tf.random.set_seed(episode)
    #     task = HoverTask(dt=dt, tracked_states=tracked_states, state_weights=state_weights, period=20, amp=20)
    #     env = Helicopter3DOF(task=task, t_max=120, dt=dt)
    #     col = CollectivePID(dt=dt, h_ref=10, derivative_gain=0.2)
    #     agent = HDPAgentTF(collective_controller=col, gamma=0.5, lr_actor=0.02, lr_critic=0.02)
    #     reward, episode_stats, weights = agent.train(env, trim_speed=trim_speed, n_updates=1, print_runtime=True)
    #     plot_stats_3dof(episode_stats, info=agent.info, results_only=True)

    #plot_neural_network_weights_2(weights)
    # q_range = np.deg2rad(np.arange(-5, 5, 0.25))
    # qerr_range = np.deg2rad(np.arange(-2, 2, 0.1))
    #
    # Z = plot_policy_function(agent, q_range, qerr_range)

