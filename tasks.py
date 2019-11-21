import numpy as np


class Task:
    """
    Base class for tracking tasks.
    """
    def __init__(self, dt=0.02):
        self.dt = dt
        self.t = 0

    def get_ref(self):
        """ Overwrite this to make a custom tracking task """
        raise NotImplementedError

    def step(self):
        """ Basic method that advances the tracking task one timestep and gets a new reference value """
        self.t += self.dt
        return self.get_ref()

    def reset(self):
        self.t = 0
        return self.get_ref()


class SimpleTrackingTask(Task):

    def __init__(self, amplitude=np.deg2rad(10), period=20):
        super().__init__()
        self.amplitude = amplitude
        self.period = period   # seconds
        self.q_ref = 0

    def get_ref(self):
        return self.amplitude * np.sin(2 * np.pi * self.t / self.period)


class HoverTask(Task):

    def __init__(self,
                 dt,
                 tracked_states=(5,),
                 state_weights=(100,),
                 period=40,
                 amp=1):
        super().__init__(dt)
        self.selected_states = tracked_states
        self.P = np.eye(7)[tracked_states, :]
        self.state_weights = np.diag(state_weights)
        self.period = period
        self.A = amp

    def get_ref(self):
        return self.A * np.pi / 180 * np.sin(2 * np.pi * self.t / self.period)

    # TODO: Create one location where the states are selected for both the task and the ACD
