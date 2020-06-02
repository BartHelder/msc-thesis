import numpy as np
import pickle


class RecursiveLeastSquares:
    """
    Incremental recursive least squares (RLS) estimator. Slightly modified from http://github.com/dave992/msc-thesis
    """
    def __init__(self, **kwargs):

        # Read kwargs
        self.state_size = kwargs['state_size']
        self.action_size = kwargs['action_size']
        self.gamma = kwargs.get('gamma', 0.8)
        self.covariance = kwargs.get('covariance', 10**4)
        self.constant = kwargs.get('constant', True)
        self.nb_vars = self.state_size + self.action_size
        if self.constant:
            self.nb_coefficients = self.state_size + self.action_size + 1
            self.constant_array = np.array([[[1]]])
        else:
            self.nb_coefficients = self.state_size + self.action_size

        # Initialize
        self.reset()

    def update(self, state, action, next_state):
        """ Update parameters and covariance. """

        state = state.reshape([-1, 1])
        action = action.reshape([-1, 1])
        next_state = next_state.reshape([-1, 1])

        if self.skip_update:
            # Store x and u
            self.x = state
            self.u = action
            self.skip_update = False
            return

        self.X[:self.state_size] = state - self.x
        self.X[self.state_size:self.nb_vars] = action - self.u
        Y = next_state - state
        # Store x and u
        self.x = state
        self.u = action

        # Error
        Y_hat = np.matmul(self.X.T, self.W).T
        error = (Y - Y_hat).T

        # Intermediate computations
        covX = np.matmul(self.cov, self.X)
        Xcov = np.matmul(self.X.T, self.cov)
        gamma_XcovX = self.gamma + np.matmul(Xcov, self.X)

        # Update weights and covariance
        self.W = self.W + np.matmul(covX, error) / gamma_XcovX
        self.cov = (self.cov - np.matmul(covX, Xcov) / gamma_XcovX) / self.gamma

    def predict(self, state, action):
        state = state.reshape([-1, 1])
        action = action.reshape([-1, 1])

        self.X[:self.state_size] = state - self.x
        self.X[self.state_size:self.nb_vars] = action - self.u
        X_next_pred = state + np.matmul(self.W.T, self.X)

        return X_next_pred

    def gradient_state(self):
        gradients = self.W[:self.state_size, :].T
        return gradients

    def gradient_action(self):
        gradient = self.W[self.state_size:self.nb_vars, :].T
        return gradient

    def reset(self):
        " Reset parameters and covariance. Check if last state is  "

        self.X = np.ones([self.nb_coefficients, 1])
        # self.W      = np.eye(self.nb_coefficients, self.state_size)
        self.W = np.zeros([self.nb_coefficients, self.state_size])
        self.cov = np.identity(self.nb_coefficients) * self.covariance

        self.x = np.zeros([self.state_size, 1])
        self.u = np.zeros([self.action_size, 1])
        self.skip_update = True

    def reset_covariance(self):
        self.cov = np.identity(self.nb_coefficients) * self.covariance

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
