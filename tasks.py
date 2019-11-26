import numpy as np


class Task:
    """
    Base class for tracking tasks.
    """
    def __init__(self, dt=0.02):
        self.dt = dt
        self.t = 0

    def get_ref(self, *args, **kwargs):
        """ Overwrite this to make a custom tracking task """
        raise NotImplementedError

    def step(self, **kwargs):
        """ Basic method that advances the tracking task one timestep and gets a new reference value """
        self.t += self.dt
        return self.get_ref(**kwargs)

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


# class HoverTask(Task):
#
#     def __init__(self,
#                  dt,
#                  tracked_states=(5,),
#                  state_weights=(100,),
#                  period=40,
#                  amp=1):
#         super().__init__(dt)
#         self.selected_states = tracked_states
#         self.P = np.eye(7)[tracked_states, :]
#         self.state_weights = np.diag(state_weights)
#         self.period = period
#         self.A = amp
#         self.move = 'sin'
#         self.corr_u = 0
#
#     # TODO: Create one location where the states are selected for both the task and the ACD
#     # def get_ref(self):
#     #     if self.t < 120:
#     #         return self.A/2 * np.pi / 180 * (np.sin(2*np.pi*self.t / self.period) + np.sin(np.pi*self.t / self.period))
#     #
#     #     elif 130 < self.t < 180:
#     #         u_err = 0 - kwargs['state'][2]
#     #
#     #         pitch_ref = -0.005 * u_err - 0.0005 * self.corr_u
#     #         self.corr_u += u_err * self.dt
#     #
#     #         return pitch_ref
#     #
#     #     else:
#     #         return 0
