import numpy as np


class IncrementalRLS:
    """Implementation of the Recursive Least Squares (RLS) filter as incremental model"""

    def __init__(self, state_size, control_size, gamma=.8):
        """Populates attributes and initializes state covariance matrix and parameter matrix"""

        # populate attributes
        self.gamma = gamma
        self.state_size = state_size
        self.control_size = control_size
        self.epsilon = None
        self.X = None
        self.delta_x_hat = None

        # initialize covariance and parameter matrix
        self.theta = np.zeros((self.state_size + self.control_size, self.state_size))
        self.Cov = np.identity(self.state_size + self.control_size)

    def update(self, delta_x, delta_u, delta_x_next):
        """Updates RLS parameters based on one sample pair"""

        # stack input data, predict, compute innovation
        self.X = np.vstack((delta_x, delta_u))
        self.delta_x_hat = np.matmul(self.X.T, self.theta).T
        self.epsilon = delta_x_next.T - self.delta_x_hat.T

        # intermediate computations
        CX = np.matmul(self.Cov, self.X)
        XC = np.matmul(self.X.T, self.Cov)
        gamma_XcovX = self.gamma + np.matmul(XC, self.X)

        # update parameter and covariance matrix
        self.theta = self.theta + np.matmul(CX, self.epsilon) / gamma_XcovX
        self.Cov = (self.Cov - np.matmul(CX, XC) / gamma_XcovX) / self.gamma

    def predict(self, delta_x, delta_u, x):
        """Predict next state using RLS parameters"""

        # extract state and input matrix estimates
        F = self.theta[:self.state_size, :].T
        G = self.theta[self.state_size:, :].T

        # return state prediction
        return x + np.matmul(F, delta_x) + np.matmul(G, delta_u)

    def get_grads(self):
        """Returns current estimates of state and input matrix"""

        # extract state and input matrix estimates
        F = self.theta[:self.state_size, :].T
        G = self.theta[self.state_size:, :].T

        return F, G

    def reset(self):
        """Resets the covariance and parameter matrix"""

        # reset covariance and parameter matrix
        self.theta = np.zeros((self.state_size + self.control_size, self.state_size))
        self.Cov = np.identity(self.state_size + self.control_size)


class RecursiveLeastSquares:

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
        return gradient[:, :, None]

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