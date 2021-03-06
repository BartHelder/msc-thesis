import torch
import torch.nn as nn

import numpy as np
import copy


class DHPCritic(nn.Module):
    """
    Torch implementation of single-layer ff neural network for DHP critic.
    """
    def __init__(self, ni, nh=8, std=0.1):
        """
        :param ni: number of inputs
        :param nh: number of hidden neurons
        :param std: standard deviation of hidden layer initialization
        """
        super(DHPCritic, self).__init__()
        self.fc1 = nn.Linear(ni, nh, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0, std=std)
        self.fc2 = nn.Linear(nh, ni, bias=False)
        nn.init.normal_(self.fc2.weight, mean=0, std=std)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class DHPActor(nn.Module):

    def __init__(self, ni, nh=8, std=0.1, scaling=None, action_network_final_layer='tanh'):
        """
        :param ni: number of inputs
        :param nh: number of hidden neurons
        :param std: standard deviation of hidden layer initialization
        :param scaling: Scaling of the output neuron
        :param action_network_final_layer: Final activation function, depends on controlled model / environment
        """

        super(DHPActor, self).__init__()
        self.fc1 = nn.Linear(ni, nh, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0, std=std)
        self.fc2 = nn.Linear(nh, 1, bias=False)
        nn.init.normal_(self.fc2.weight, mean=0, std=std)
        if scaling is not None:
            self.scale = np.deg2rad(scaling)
        else:
            self.scale = 1
        # The 6dof model takes inputs [0, 1] while the 3dof model takes them in degrees of control deflection.
        # Therefore, we need to switch the final layer based on the model used:
        if action_network_final_layer == 'tanh':
            self.final_activation = torch.tanh
        elif action_network_final_layer == 'sigmoid':
            self.final_activation = torch.sigmoid
        else:
            raise NotImplementedError(f"Activation function {action_network_final_layer} not supported. ")

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.final_activation(self.fc2(x))
        x = self.scale * x
        return x


class DHPAgent:
    """
    An agent is a container for three neural networks (actor, critic, and target critic) as well as a reward
    function. Contains an interface for interacting with the environment (get_action) and a RLS model of the
    environment (get_transition_matrices). The Incremental Dual Heuristic Programming (IDHP) update mechanism
    is roughly according to Heyer, Kroezen, Van Kampen (2020), except that the target critic estimate is also used in
    the actor update.
    """
    def __init__(self, control_channel: str, discount_factor,
                 n_hidden_actor: int, nn_stdev_actor, learning_rate_actor, action_scaling, action_network_final_layer,
                 n_hidden_critic: int, nn_stdev_critic, learning_rate_critic, tau_target_critic,
                 tracked_state: int, ac_states: list,
                 reward_weight=1):
        """
        :param control_channel: three-letter code of the control channel. Choose from: col, lon, lat, ped
        :param discount_factor
        :param n_hidden_actor: Number of hidden neurons in the actor network
        :param nn_stdev_actor: Standard deviation of the hidden layer initialization of the actor
        :param learning_rate_actor: Learning rate (alpha) of the actor
        :param action_scaling: Scaling of the output of the actor in degrees, should correspond roughly to real actuator limits
        :param action_network_final_layer: Activation used for final layer of action networks, depends on environment
        :param n_hidden_critic: Number of hidden neurons in the critic network
        :param nn_stdev_critic: Standard deviation of the hidden layer initialization of the critic
        :param learning_rate_critic: Learning rate (alpha) of the critic
        :param tau_target_critic: Time delay / update speed of the target critic.
                                   tau=1 means they are the same, 0<tau<1 means a lagged target critic is used
        :param tracked_state: Primary state the ACD attempts to track. The tracking error of this state is an input
        :param ac_states: Auxiliary states also fed into the ACD
        :param reward_weight:
        """
        self.control_channel_str = control_channel
        self.control_channel_mapping = {'col': 0, 'lon': 1, 'lat': 2, 'ped': 3}
        self.control_channel_num = self.control_channel_mapping[control_channel]

        self.tracked_state = tracked_state
        self.ac_states = ac_states
        self.n_inputs = len(ac_states)+1

        self.actor = DHPActor(ni=self.n_inputs, nh=n_hidden_actor, std=nn_stdev_actor, scaling=action_scaling,
                              action_network_final_layer=action_network_final_layer)
        self.critic = DHPCritic(ni=self.n_inputs, nh=n_hidden_critic, std=nn_stdev_critic)
        self.target_critic = copy.deepcopy(self.critic)

        self.networks = [self.actor, self.critic, self.target_critic]

        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma = discount_factor
        self.tau_target_critic = tau_target_critic
        self.reward_weight = reward_weight

    def get_action(self, state, ref):
        augmented_state = self.augment_state(state, ref)
        with torch.no_grad():
            action = self.actor.forward(augmented_state)
        return action.data

    def get_reward(self, next_state, ref):
        dr_ds = torch.zeros((1, len(self.ac_states) + 1))
        tracking_error = next_state[self.tracked_state] - ref[self.tracked_state]  # r_{t+1} = f(s_{t+1}, sRef_{t})
        reward = -tracking_error**2 * self.reward_weight
        dr_ds[:, -1] = -2 * tracking_error * self.reward_weight
        return reward, dr_ds

    def get_transition_matrices(self, rls_estimator):
        """
        Extract the correct submatrices from the incremental RLS system matrices, for use in updating the actor and
        critic.
        :param rls_estimator: RLS estimator instance
        :return: torch tensors F and G, containing only the elements used by the ACD
        """
        state_indices = self.ac_states + [self.tracked_state]
        F = rls_estimator.gradient_state()[np.ix_(state_indices, state_indices)]
        G = rls_estimator.gradient_action()[np.ix_(state_indices, [self.control_channel_num])]
        return torch.tensor(F, dtype=torch.float), torch.tensor(G, dtype=torch.float)

    def augment_state(self, state, reference):
        """
        Transform observation and reference into the augmented state form,
        :param state:
        :param reference:
        :return:
        """
        augmented_state = [state[x] for x in self.ac_states] + [state[self.tracked_state] - reference[self.tracked_state]]
        return torch.tensor(augmented_state, requires_grad=True)

    def update_networks(self, state, next_state, ref, next_ref, dr_ds, F, G):
        """
        Update the actor, critic and target critic model by doing forward and backward passes through the respective
        neural networks according to the Dual Heuristic Dynamic Programming (DHP) strategy.
        :param state:
        :param next_state:
        :param ref:
        :param next_ref:
        :param dr_ds:
        :param F:
        :param G:
        :return:
        """
        augmented_state = self.augment_state(state, ref)
        next_augmented_state = self.augment_state(next_state, next_ref)

        # Forward passes...
        action = self.actor.forward(augmented_state)
        lambda_t1 = self.critic.forward(augmented_state)
        lambda_t2 = self.target_critic.forward(next_augmented_state)

        # Backpropagate raw action through actor network
        action.backward()
        da_ds = augmented_state.grad

        # From DHP definition:
        target = dr_ds + self.gamma * lambda_t2
        error_critic = lambda_t1 - target.mm(F + G.mm(da_ds.unsqueeze(0)))

        # Backpropagate error_critic through critic network and update weights
        lambda_t1.backward(error_critic.squeeze())
        # Make sure these calculations don't affect the actual gradients by wrapping them in no_grad()
        with torch.no_grad():
            for wa, wc in zip(self.actor.parameters(), self.critic.parameters()):
                # .sub_() is in-place subtraction (NOT SUBSTITUTION!!!) - fast en memory-efficient
                wa.data.sub_(wa.grad.data * (-target.mm(G).squeeze(dim=0)) * self.learning_rate_actor)
                wc.data.sub_(wc.grad.data * self.learning_rate_critic)
        # In PyTorch, gradients accumulate rather than overwrite, so after updating they must be zeroed:
        self.critic.zero_grad()
        self.actor.zero_grad()
        self.target_critic.zero_grad()  # I don't think these have a value inside of them but just to be sure...

        # Update target network - copy_() is a fast and memory-unintensive value overwrite
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau_target_critic * param.data + (1.0 - self.tau_target_critic) * target_param.data)

    def save(self, path):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
