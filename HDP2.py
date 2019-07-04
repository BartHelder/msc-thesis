import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
from plotting import plot_stats
import pandas as pd
from time import sleep
import gym
import matplotlib.pyplot as plt
import sys
import itertools
from heli_simple import SimpleHelicopter

tf.random.set_seed(123)

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.h1_p = kl.Dense(128,  activation='relu')
        self.h1_v = kl.Dense(128, activation='relu')
        self.policy = kl.Dense(1, name='policy')
        self.value = kl.Dense(1, name='value')

    def call(self, inputs):
        # input goes via numpy array, so convert to Tensor first
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)

        # separate hidden layers from the same input tensor
        p1 = self.h1_p(x)
        v1 = self.h1_v(x)

        out = self.policy(p1)

        return tf.clip_by_value(out, -np.deg2rad(5), np.deg2rad(5)), self.value(v1)

    def action_value(self, obs):
        # executes call() under the hood
        action, value = self.predict(obs)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

class HDPAgent:

    def __init__(self, model):
        self.params = {
            'value': 0.5,
            'entropy': 0.0001,
            'gamma': 0.95}
        self.model = model
        self.model.compile(
            optimizer=ko.Adam(lr=0.0001),
            # define separate losses for policy logits and value estimate
            loss=[self._actor_loss, self._critic_loss]
        )

    def train(self, env, batch_size=32, updates=20000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values, = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + env.observation_space.shape)
        # training loop: collect samples, send to optimizer, repeat updates times
        episode_rewards = [0.0]

        # Initialize environment and tracking task
        next_obs = env.reset()
        task = TrackingTask(dt=env.dt)
        q_ref = task.reset()
        n_episodes = 1

        for update in range(updates):

            if (update+1) % 1000 == 0:
                print("Update # " + str(update+1), "\t", "Last episode reward: " + str(episode_rewards[-2]))
                self.test(env)
            for step in range(batch_size):
                # storage helpers for a single batch of data
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                # Make a step in the environment
                next_obs, rewards[step], dones[step], _ = env.step(actions[step], q_ref)
                q_ref = task.step()

                # Add rewards to storage
                episode_rewards[-1] += rewards[step]

                if task.t > env.max_episode_length:
                    dones[step] = True

                if dones[step]:

                    # Reset environment + task
                    episode_rewards.append(0.0)
                    next_obs = env.reset()
                    q_ref = task.reset()
                    n_episodes += 1

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through the same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # perform a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])

        print(" Total episodes run: ", n_episodes)
        return episode_rewards

    def test(self, env, max_steps=None, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        task = TrackingTask(dt=env.dt)
        q_ref = task.reset()
        action = 0
        stats = []
        while not done:
            stats.append({'t': task.t, 'q': obs[0], 'q_ref': q_ref, 'u': action})

            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action[0], q_ref)
            q_ref = task.step()
            ep_reward += reward
            max_episode_length = env.max_episode_length if max_steps is None else max_steps
            if task.t > max_episode_length:
                done=True
        if render:
            df = pd.DataFrame(stats)
            plot_stats(df, 'final')

        return ep_reward

    def _critic_loss(self, td_target, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(td_target, value)

    def _actor_loss(self, value, value_min=0):

        return self.params['value'] * kls.mean_squared_error(value, value_min)

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estmate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are caluclated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages


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
    env = SimpleHelicopter(tau=0.05, k_beta=0, name='poep')
    model = Model()
    agent = HDPAgent(model)
    np.random.seed(1)
    print("Before training: %f" % agent.test(env, render=True))
    print("Starting training phase...")
    rewards_history = agent.train(env)
    print("Finished training, testing...")
    print("After training: %f" % agent.test(env, max_steps=80, render=True))

    env = SimpleHelicopter(tau=0.25, k_beta=1000)
    agent.test(env, max_steps=80, render=True)