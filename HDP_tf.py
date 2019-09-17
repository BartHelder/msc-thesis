import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
from plotting import plot_stats
import pandas as pd
from collections import deque
from heli_simple import SimpleHelicopter


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__('mlp_policy')
        self.h1_p = kl.Dense(6,  activation='tanh')
        self.h1_v = kl.Dense(6, activation='tanh')
        self.policy = kl.Dense(1, name='policy')
        self.value = kl.Dense(1, name='value')

    def call(self, inputs):
        # input goes via numpy array, so convert to Tensor first
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)

        # separate hidden layers from the same input tensor
        p1 = self.h1_p(x)
        v1 = self.h1_v(x)

        return self.policy(p1), self.value(v1)

    def action_value(self, obs):
        # executes call() under the hood
        action, value = self.predict(obs)

        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

def scaled_tanh(input, scale=np.deg2rad(5)):
    return tf.keras.backend.tanh(input) * scale


class nn_actor(tf.keras.Model):

    def __init__(self):
        super().__init__('mlp_policy')
        self.h1_p = kl.Dense(6, activation='tanh')
        self.policy = kl.Dense(1, activation='linear')

    def call(self, input):
        x = tf.convert_to_tensor(input, dtype=tf.float32)
        p1 = self.h1_p(x)
        return self.policy(p1)

    def action(self, obs):
        action = self.predict(obs)
        return np.squeeze(action, axis=-1)
        # return action


class nn_critic(tf.keras.Model):

    def __init__(self):
        super().__init__('mlp_critic')
        self.h1_c = kl.Dense(6, activation='tanh')
        self.critic = kl.Dense(1, name='value')

    def call(self, input):
        x = tf.convert_to_tensor(input, dtype=tf.float32)
        v1 = self.h1_c(x)
        return self.critic(v1)

    def value(self, obs):
        value = self.predict(obs)
        return np.squeeze(value, axis=-1)


nn_actor2 = tf.keras.Sequential([
    kl.Dense(6,  activation='tanh'),
    kl.Dense(1, activation=scaled_tanh)
])


nn_critic2 = tf.keras.Sequential([
    kl.Dense(6,  activation='tanh'),
    kl.Dense(1, activation='linear')
])



optimizer_ac = ko.SGD(lr=0.001)
optimizer_critic = ko.SGD(lr=0.01)
loss_fn = kls.mean_squared_error

loss_metric = tf.keras.metrics.Mean(name='train_loss')


class HDPAgent:

    def __init__(self, actor=nn_actor2, critic=nn_critic2):
        self.gamma = 0.99
        self.actor = actor
        self.critic = critic


    def train(self,
              env,
              n_episodes=500,
              n_updates_critic=5,
              n_updates_actor=5,
              threshold_loss_critic=0.05,
              threshold_loss_actor=0.01):

        # updates = int(env.max_episode_length / (env.dt * batch_size))
        # # storage helpers for a single batch of data
        # actions = np.empty((batch_size,), dtype=np.int32)
        # rewards, dones, values, = np.empty((3, batch_size))
        # observations = np.empty((batch_size,) + env.observation_space.shape)

        # training loop: collect samples, send to optimizer, repeat updates times
        episode_rewards = [0.0]
        task = TrackingTask(dt=env.dt)

        for n_episode in range(n_episodes):
            if (n_episode + 1) % 10 == 0:
                print("Update # " + str(n_episode + 1))
                self.test(env, render=True)

            # Initialize environment and tracking task
            observation = env.reset()
            action = self.actor(observation[None, :])
            value = self.critic(observation[None, :])
            next_observation, reward, done = env.step(action.numpy()[0][0])

            previous_value = value
            observation = next_observation

            # Repeat (for each step t of an episode)
            for step in range(int(env.episode_ticks)):

                action = self.actor(observation[None, :])
                value = self.critic(observation[None, :])
                next_observation, reward, done = env.step(action.numpy()[0][0])

                td_target = reward + self.gamma * value - previous_value

                for j in range(2):
                    with tf.GradientTape() as tape:
                        loss_critic = 1/2 * kls.mean_squared_error(td_target, self.critic(observation[None, :]))

                    dEc_dwc = tape.gradient(loss_critic, self.critic.trainable_variables)
                    optimizer_critic.apply_gradients(zip(dEc_dwc, self.critic.trainable_variables))

                    with tf.GradientTape as tape2:
                        loss_actor = 1/2 * kls.mean_squared_error()

                    # dEa_dea =
                    # dea_dV = 1
                    # dV_ds =
                    #ds_da = np.array([-0.04062, 0.0001293, -0.04062]).reshape((3, 1))
                    #da_dwa =


                # 1. Update critic
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(no)
                    act = self.actor(observation[None, :])
                    tape.watch(act)
                    next_value = self.critic(no)
                    tape.watch(next_value)
                    td_target = reward + self.gamma * next_value
                    tape.watch(td_target)


                dEc_dwc = tape.gradient(loss1, self.critic.trainable_variables)
                optimizer_critic.apply_gradients(zip(dEc_dwc, self.critic.trainable_variables))

                # 2. Update actor

                dV_ds = tape.gradient(next_value, no)
                da_dwa = tape.gradient(act, self.actor.trainable_variables)
                ds_da = np.array([-0.04062, 0.0001293, -0.04062]).reshape((3, 1))
                dEa_da = tf.matmul(tf.multiply(next_value, dV_ds), ds_da)
                dEa_dwa = [tf.multiply(dEa_da, i) for i in da_dwa]
                dEa_dwa[1] = tf.squeeze(dEa_dwa[1])
                dEa_dwa[3] = tf.reshape(dEa_dwa[3], (1,))
                optimizer_ac.apply_gradients(zip(dEa_dwa, self.actor.trainable_variables))


                observation = next_observation
                value = next_value

            print('episode done')
            self.test(env, max_steps=80)
        return episode_rewards

    def test(self, env, max_steps=None, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        action = 0
        stats = []
        while not done:
            stats.append({'t': env.task.t, 'q': obs[0], 'q_ref': env.task.get_q_ref(), 'a1': obs[1], 'u': action})
            action = self.actor(obs[None, :])
            obs, reward, done = env.step(action.numpy()[0][0], env.task.get_q_ref())
            ep_reward += reward
            max_episode_length = env.max_episode_length if max_steps is None else max_steps
            if env.task.t > max_episode_length:
                done=True
        if render:
            df = pd.DataFrame(stats)
            plot_stats(df, 'final')

        return ep_reward

    def _train_step(self, reward, value, observation, next_observation):

        # 1. Update critic
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(no)
            act = self.actor(observation[None, :])
            tape.watch(act)
            next_value = self.critic(no)
            tape.watch(next_value)
            td_target = reward + self.gamma * next_value
            tape.watch(td_target)
            loss1 = kls.mean_squared_error(td_target, value)

        dEc_dwc = tape.gradient(loss1, self.critic.trainable_variables)
        optimizer_critic.apply_gradients(zip(dEc_dwc, self.critic.trainable_variables))

        # 2. Update actor

        dV_ds = tape.gradient(next_value, no)
        da_dwa = tape.gradient(act, self.actor.trainable_variables)
        ds_da = np.array([-env.th_iy * env.dt, env.dt / env.tau, -env.th_iy * env.dt]).reshape((3, 1))
        dEa_da = tf.matmul(tf.multiply(next_value, dV_ds), ds_da)
        dEa_dwa = [tf.multiply(dEa_da, i) for i in da_dwa]
        dEa_dwa[1] = tf.squeeze(dEa_dwa[1])
        dEa_dwa[3] = tf.reshape(dEa_dwa[3], (1,))
        optimizer_ac.apply_gradients(zip(dEa_dwa, self.actor.trainable_variables))

        del tape

        return loss1



class TrackingTask:

    def __init__(self,
                 amplitude=np.deg2rad(3),
                 period=20,
                 dt=0.02):
        self.amplitude = amplitude
        self.period = period   # seconds
        self.dt = dt
        self.t = 0
        self.q_ref = 0

    def get_q_ref(self):
        return self.amplitude * np.sin(2 * np.pi * self.t / self.period)

    def step(self):
        self.t += self.dt
        return self.get_q_ref()

    def reset(self):
        self.t = 0
        return self.get_q_ref()


if __name__ == '__main__':

    task = TrackingTask(period=40)
    env = SimpleHelicopter(tau=0.05, k_beta=400000, task=task, name='poep')
    agent = HDPAgent()
    np.random.seed()
    print("Before training: %f out of 200" % agent.test(env, render=True))
    print("Starting training phase...")
    rewards_history = agent.train(env)
    print("Finished training, testing...")
    print("After training: %f out of 200" % agent.test(env, max_steps=80, render=True))

    env2 = SimpleHelicopter(tau=0.25, k_beta=1000)
    agent.test(env2, max_steps=80, render=True)
