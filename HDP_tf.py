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
import itertools

from heli_models import Helicopter3DOF
from plotting import plot_neural_network_weights_2, plot_stats_3dof, plot_policy_function
from model import RecursiveLeastSquares
tf.keras.backend.set_floatx('float64')


def scaled_tanh(scale, x):
    return tf.tanh(x) * scale


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

    def call(self, x):
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

        self.optimizer = ko.SGD(lr=self.lr_actor, nesterov=bool(conf["use_nesterov_momentum"]), momentum=conf["momentum"])
        # self.optimizer_actor_cyclic = ko.SGD(lr=self.lr_actor,
        #                                      nesterov=bool(conf["use_nesterov_momentum"]),
        #                                      momentum=conf["momentum"])
        # self.optimizer_actor_collective = ko.SGD(lr=self.lr_actor,
        #                                          nesterov=bool(conf["use_nesterov_momentum"]),
        #                                          momentum=conf["momentum"])
        # self.optimizer_critic = ko.SGD(lr=self.lr_critic,
        #                                nesterov=bool(conf["use_nesterov_momentum"]),
        #                                momentum=conf["momentum"])
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

            return tf.constant([[obs[x] for x in actor_states] + [tracking_error]])

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

    #@tf.function
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
            self.optimizer.apply_gradients(zip(gradient_critic, self.critic.trainable_variables))

            # Actor gradient through the critic and environment model: dEa/dwa = V * dV/ds * ds/da * da/dwa
            dV_ds = tape.gradient(value, s_aug2)
            scale = tf.squeeze(tf.multiply(value, tf.matmul(dV_ds, ds_da)))

            #da1_dwa = tape.gradient(cyclic, self.actor_cyclic.trainable_variables)
            da2_dwa = tape.gradient(collective, self.actor_collective.trainable_variables)
            #gradient_actor1 = [tf.multiply(scale, x) for x in da1_dwa]
            gradient_actor2 = [tf.multiply(scale, x) for x in da2_dwa]

            #self.optimizer_actor_cyclic.apply_gradients(zip(gradient_actor1, self.actor_cyclic.trainable_variables))
            self.optimizer.apply_gradients(zip(gradient_actor2, self.actor_collective.trainable_variables))

    def save(self, path="saved_models/"+datetime.date.today().isoformat()+"/"):
        self.actor_cyclic.save_weights(path + "actor", save_format='tf')
        self.critic.save_weights(path + "critic", save_format='tf')

    def load(self, path):
        self.actor_cyclic.load_weights(path + "actor")
        self.critic.load_weights(path + "critic")


class Agent:

    def __init__(self,
                 config,
                 actor,
                 actor_kwargs,
                 control_channel
                 ):

        with open(config, "r") as f:
            c = json.load(f)
        conf = c["agent"][control_channel]
        self.control_channel = control_channel
        initializer = tf.initializers.TruncatedNormal(mean=0.0, stddev=conf["weights_stddev"])
        self.actor = actor(n_hidden=conf["n_hidden"],
                           initializer=initializer,
                           **actor_kwargs)
        self.critic = TFCritic(n_hidden=conf["n_hidden"], initializer=initializer)
        self.optimizer_actor = ko.SGD(lr=conf["lr_actor"],
                                nesterov=bool(conf["use_nesterov_momentum"]),
                                momentum=conf["momentum"])
        self.optimizer_critic = ko.SGD(lr=conf["lr_critic"],
                                nesterov=bool(conf["use_nesterov_momentum"]),
                                momentum=conf["momentum"])
        self.ds_da = None
        self.loss_object = kls.MeanSquaredError()
        self.tracked_state = conf["tracked_state"]
        self.actor_critic_states = conf["actor_critic_states"]
        self.reward_weight = conf["reward_weight"]
        self.gamma = conf["gamma"]
        self.info = {'n_hidden': conf["n_hidden"],
                     'gamma': conf["gamma"],
                     'lr_actor': conf["lr_actor"],
                     'lr_critic': conf["lr_critic"]}

    def get_action(self, augmented_state):
        pass

    def augment_state(self, observation, reference):

        tracking_error = reference[self.tracked_state] - observation[self.tracked_state]
        return tf.constant([[observation[x] for x in self.actor_critic_states] + [tracking_error]])

    def get_reward(self, observation, reference):
        tracking_error = reference[self.tracked_state] - observation[self.tracked_state]
        reward = -tracking_error**2 * self.reward_weight
        reward = np.clip(reward, -5, 0)
        return reward

    def set_ds_da(self, rls_model):
        mapping = {'collective': 0,
                   'cyclic_lon': 1,
                   'cyclic_lat': 2,
                   'directional': 3}

        ds_da = rls_model.gradient_action()[:, mapping[self.control_channel]]
        state_transition = ds_da[self.actor_critic_states]
        tracking_error = -ds_da[self.tracked_state].reshape((1, 1))  # if the state increases, tracking error decreases proportionally
        dsda = np.append(state_transition, tracking_error, axis=0)
        self.ds_da = tf.constant(dsda)

        return dsda

    #@tf.function
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
            #print(self.control_channel)
            # print(max(abs(gradient_actor[0].numpy().ravel())))
            self.optimizer_actor.apply_gradients(zip(gradient_actor, self.actor.trainable_variables))

    def get_weights(self):
        weight_stats = {'wci': self.critic.trainable_weights[0].numpy().ravel(),
                        'wco': self.critic.trainable_weights[1].numpy().ravel(),
                        'wai': self.actor.trainable_weights[0].numpy().ravel(),
                        'wao': self.actor.trainable_weights[1].numpy().ravel()
                        }

        return weight_stats

    def save(self, name, folder="saved_models/"+datetime.date.today().isoformat()+"/"):
        self.actor.save_weights(folder + "actor_" + name, save_format='tf')
        self.critic.save_weights(folder + "critic_" + name, save_format='tf')

    def load(self, name, folder):
        self.actor.load_weights(folder + "actor_" + name)
        self.critic.load_weights(folder + "critic_" + name)


def train_save_pitch(seed, save_path, config_path="config.json"):

    with open(config_path, 'r') as f:
        config = json.load(f)

    dt = config["dt"]  # s
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

    tf.random.set_seed(3)
    cfp = "config_3dof.json"

    env = Helicopter3DOF()
    env.setup_from_config(task="sinusoid", config_path=cfp)
    observation, trim_actions = env.reset(v_initial=20)

    rls_kwargs = {'state_size': 7, 'action_size': 2, 'gamma': 1, 'covariance': 10**8, 'constant': False}
    RLS = RecursiveLeastSquares(**rls_kwargs)
    CollectiveAgent = Agent(cfp, actor=TFActor3DOF, actor_kwargs={'offset': 0, 'action_scaling': np.rad2deg(trim_actions[0])}, control_channel='collective')
    CollectiveAgent.set_ds_da(RLS)
    CyclicAgent = Agent(cfp, actor=TFActor3DOF, actor_kwargs={'offset': 0, 'action_scaling': 10}, control_channel="cyclic_lon")
    CyclicAgent.set_ds_da(RLS)
    agents = (CollectiveAgent, CyclicAgent)
    stats = []
    reward = [None, None]
    weight_stats = {'t': [], 'wci': [], 'wco': [], 'wai': [], 'wao': []}
    rls_stats = {'t': [0],
                 'wa_col': [RLS.gradient_action()[:6, 0].ravel().copy()],
                 'wa_cyc': [RLS.gradient_action()[:6, 1].ravel().copy()],
                 'ws': [RLS.gradient_state().ravel().copy()]}

    excitation = np.zeros((1000, 2))
    # for j in range(50):
    #     excitation[j, 0] = j / 50
    # for j in range(50, 150, 1):
    #     excitation[j, 0] = 2 - j / 50
    # for j in range(150, 250, 1):
    #     excitation[j, 0] = -4 + j / 50
    # for j in range(250, 350, 1):
    #     excitation[j, 0] = 6 - j / 50
    # for j in range(350, 400, 1):
    #     excitation[j, 0] = -8 + j / 50
    #
    # for j in range(50):
    #     excitation[j + 400, 1] = j / 50
    # for j in range(50, 150, 1):
    #     excitation[j + 400, 1] = 2 - j / 50
    # for j in range(150, 250, 1):
    #     excitation[j + 400, 1] = -4 + j / 50
    # for j in range(250, 350, 1):
    #     excitation[j + 400, 1] = 6 - j / 50
    # for j in range(350, 400, 1):
    #     excitation[j + 400, 1] = -8 + j / 50

    for j in range(400):
        excitation[j, 0] = -np.sin(np.pi*j/50)
        excitation[j+400, 1] = np.sin(2*np.pi*j/50) * 2

    excitation = np.deg2rad(excitation)
    excitation_phase = True
    done = False
    step = 0
    while not done:

        # Get new reference
        reference = env.get_ref()

        # Augment state with tracking errors
        augmented_states = (CollectiveAgent.augment_state(observation, reference),
                            CyclicAgent.augment_state(observation, reference))

        # Get actions from actors
        actions = np.array([CollectiveAgent.actor(augmented_states[0]).numpy().squeeze(),
                            CyclicAgent.actor(augmented_states[1]).numpy().squeeze()])

        # In excitation phase add values to actions
        if excitation_phase:
            actions += excitation[step]

        actions = actions + trim_actions
        actions = np.clip(actions, np.deg2rad([0, -15]), np.deg2rad([10, 15]))
        # Take step in the environment
        next_observation, _, done = env.step(actions)

        # Update RLS model
        RLS.update(state=observation, action=actions, next_state=next_observation)

        # Update action gradients
        CollectiveAgent.set_ds_da(RLS)
        CyclicAgent.set_ds_da(RLS)

        # Get rewards, update actor and critic networks
        for agent, count in zip(agents, itertools.count()):
            reward[count] = agent.get_reward(next_observation, reference)
            next_augmented_state = agent.augment_state(next_observation, reference)
            td_target = reward[count] + agent.gamma * agent.critic(next_augmented_state)
            agent.update_networks(td_target, augmented_states[count], n_updates=1)
            if count == 1:
                break

        # Log data
        stats.append({'t': env.t,
                      'x': observation[0],
                      'z': observation[1],
                      'u': observation[2],
                      'w': observation[3],
                      'theta': observation[4],
                      'q': observation[5],
                      'reference': env.get_ref(),
                      'collective': actions[0],
                      'cyclic': actions[1],
                      'r1': reward[0],
                      'r2': reward[1]})

        rls_stats['t'].append(env.t)
        rls_stats['ws'].append(RLS.gradient_state().ravel())
        rls_stats['wa_col'].append(RLS.gradient_action()[:6, 0].ravel())
        rls_stats['wa_cyc'].append(RLS.gradient_action()[:6, 1].ravel())

        if env.t > 16:
            excitation_phase = False
        if env.t > 100:
            done = True

        # Next step..
        observation = next_observation
        step += 1
    stats = pd.DataFrame(stats)
    plot_stats_3dof(stats)

    wa_col = pd.DataFrame(data=rls_stats['wa_col'], index=rls_stats['t'], columns = ['ex', 'z', 'u', 'w', 'pitch', 'q'])
    wa_cyc = pd.DataFrame(data=rls_stats['wa_cyc'], index=rls_stats['t'], columns = ['ex', 'z', 'u', 'w', 'pitch', 'q'])
    wa_col = wa_col.drop(columns=['ex', 'u', 'pitch', 'q'])
    wa_cyc = wa_cyc.drop(columns=['ex', 'z', 'u', 'w'])
    ws = pd.DataFrame(data=rls_stats['ws'], index=rls_stats['t'])
    from matplotlib import pyplot as plt
    import seaborn as sns

    plt.figure()
    sns.lineplot(data=wa_col, dashes=False, legend='full', palette=sns.color_palette("hls", len(wa_col.columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Gradient size [-]')
    plt.title('iRLS Collective gradients')
    plt.show()

    plt.figure()
    sns.lineplot(data=wa_cyc, dashes=True, legend='full', palette=sns.color_palette("hls", len(wa_cyc.columns)))
    plt.xlabel('Time [s]')
    plt.ylabel('Gradient size [-]')
    plt.title('iRLS Cyclic gradients')
    plt.show()
