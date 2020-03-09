import numpy as np
import json


class CollectivePID3DOF:

    def __init__(self, h_ref, dt, proportional_gain=2, integral_gain=0.2, derivative_gain=0.1):
        self.h_ref = h_ref
        self.hdot_corr = 0
        self.hdot_err = 0
        self.dt = dt
        self.Kp = proportional_gain
        self.Ki = integral_gain
        self.Kd = derivative_gain
        self.state_indices = {'z': 1, 'u': 2, 'w': 3, 'theta': 4, 'phi': None}

    def __call__(self, obs):

        hdot_ref = self.Kd * (self.h_ref - -obs[1])
        hdot = (obs[self.state_indices['u']] * np.sin(obs[self.state_indices['theta']])
                - obs[self.state_indices['w']] * np.cos(obs[self.state_indices['theta']]))
        self.hdot_err = (hdot_ref - hdot)
        collective = np.deg2rad(5 + self.Kp * self.hdot_err + self.Ki * self.hdot_corr)
        self.hdot_corr += self.dt * self.hdot_err
        return np.clip(collective, 0, np.deg2rad(10))


class CollectivePID6DOF(CollectivePID3DOF):
    def __init__(self, col_trim, h_ref, dt, proportional_gain=0.02, integral_gain=0.002, derivative_gain=0.1):
        super().__init__(h_ref, dt, proportional_gain, integral_gain, derivative_gain)
        self.col_trim = col_trim

    def __call__(self, obs):
        u = obs[0]
        v = obs[1]
        w = obs[2]
        phi = obs[6]
        theta = obs[7]
        z = obs[11]
        h = -z

        hdot_ref = self.Kd * (self.h_ref - h)
        hdot = u * np.sin(theta) - (v * np.sin(phi) + w * np.cos(phi)) * np.cos(theta)
        hdot_err = (hdot_ref - hdot)
        collective = self.col_trim + self.Kp * hdot_err + self.Ki * self.hdot_corr
        self.hdot_corr += self.dt * hdot_err
        return np.clip(collective, 0, 1)


class LatPedPID:
    def __init__(self, phi_trim, lat_trim, pedal_trim, dt, gains_dict):
        self.dt = dt
        self.gains = np.array([gains_dict["Ky"], gains_dict["Ky_int"], gains_dict["Ky_dot"],
                               np.rad2deg(gains_dict["Kphi"]), np.rad2deg(gains_dict["Kphi"]), np.rad2deg(gains_dict["Kp"]),
                               np.rad2deg(gains_dict["Kpsi"]), np.rad2deg(gains_dict["Kpsi_int"]), np.rad2deg(gains_dict["Kr"])])
        self.phi_trim = phi_trim
        self.lat_trim = lat_trim
        self.pedal_trim = pedal_trim
        self.y_req = 0
        self.y_req_int = 0
        self.phi_int = 0
        self.psi_int = 0

    def __call__(self, obs):
        Ky, Ky_int, Ky_dot, Kphi, Kphi_int, Kp, Kpsi, Kpsi_int, Kr = self.gains
        u = obs[0]
        v = obs[1]
        w = obs[2]
        p = obs[3]
        r = obs[5]
        phi = obs[6]
        theta = obs[7]
        psi = obs[8]
        y = obs[10]
        ydot = (u * np.cos(theta) + (v * np.sin(phi) + w * np.cos(phi)) * np.sin(theta)) * np.sin(psi) \
            + (v * np.cos(phi) - w * np.sin(phi)) * np.cos(psi)
        y_error = self.y_req - y
        phi_req = self.phi_trim + Ky * y_error + Ky_int * self.y_req_int + Ky_dot * ydot
        psi_req = 0
        phi_error = (phi_req - phi)
        psi_error = (psi_req - psi)

        lat = np.clip(self.lat_trim + (Kphi * phi_error + Kphi_int * self.phi_int + Kp * p)/100, 0, 1)
        ped = np.clip(self.pedal_trim + (Kpsi * psi_error + Kpsi_int * self.psi_int + Kr * r)/100, 0, 1)

        if 0.01 < lat < 0.99:
            self.y_req_int += y_error * self.dt
            self.phi_int += phi_error * self.dt
        #self.psi_int += psi_error * self.dt

        return lat, ped








