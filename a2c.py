import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import gym.envs.classic_control.cartpole


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # samples random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        self.hidden1 = kl.Dense(6, activation='sigmoid')
        self.hidden2 = kl.Dense(6, activation='sigmoid')
        self.value = kl.Dense(1, name='value')
        # logits are non-normalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # input goes via numpy array, so convert to Tensor first
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.params = {
            'value': 0.5,
            'entropy': 0.0001,
            'gamma': 0.99}
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.007),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )

    def train(self, env, batch_size=32, updates=500):
        # storage helpers for a single batch of data
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values, = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + env.observation_space.shape)
        # training loop: collect samples, send to optimizer, repeat updates times
        episode_rewards = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_size):
                # storage helpers for a single batch of data
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                episode_rewards[-1] += rewards[step]
                if dones[step]:
                    episode_rewards.append(0.0)
                    next_obs = env.reset()

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through the same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # perform a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])

        return episode_rewards

    def test(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

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

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through the same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse cateogrical CE loss obj that supports sample_weight arg on call()
        # from_logits arg ensures transofmration into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by the policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy']*entropy_loss


if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    model = Model(num_actions=env.action_space.n)

    agent = A2CAgent(model)
    print("Before training: %d out of 200" % agent.test(env))
    print("Starting training phase...")
    rewards_history = agent.train(env)
    print("Finished training, testing...")
    print("After training: %d out of 200" % agent.test(env))
