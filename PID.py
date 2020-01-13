import numpy as np
import json

class PID():
    "PID Controller"

    def __init__(self, Kp, Ki=0, Kd=0, dt=1 / 100):

        # Set input parameters
        self.dt = dt
        self.set_gains(float(Kp), float(Ki), float(Kd))

        # Set initial PID values
        self.P = np.array([0.0])
        self.I = np.array([0.0])
        self.D = np.array([0.0])
        self.last_error = np.array([0.0])

    def __call__(self, error):
        # Proportional Term
        self.P = self.Kp * error

        # Integral Term
        self.I = self._I(error)

        # Derivative Term
        self.D = self._D(error)
        self.last_error = error

        return self.P + self.I + self.D

    def set_gains(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        if Ki == 0.0:
            self._I = lambda error: 0.0
        else:
            self._I = lambda error: self.I + self.Ki * error * self.dt

        if Kd == 0.0:
            self._D = lambda error: 0.0
        else:
            self._D = lambda error: -self.Kd * (error - self.last_error) / self.dt



class CollectivePID:

    def __init__(self, h_ref=25, dt=0.01, proportional_gain=2, integral_gain=0.2, derivative_gain=0.1):
        self.h_ref = h_ref
        self.hdot_corr = 0
        self.hdot_err = 0
        self.dt = dt
        self.Kp = proportional_gain
        self.Ki = integral_gain
        self.Kd = derivative_gain

    def __call__(self, obs):

        hdot_ref = self.Kd * (self.h_ref - -obs[1])
        hdot = (obs[2] * np.sin(obs[4]) - obs[3] * np.cos(obs[4]))
        self.hdot_err = (hdot_ref - hdot)
        collective = np.deg2rad(5 + self.Kp * self.hdot_err + self.Ki * self.hdot_corr)

        return collective

    def increment_hdot_error(self):
        self.hdot_corr += self.dt * self.hdot_err


class LatPedPID:
    def __init__(self, phi_trim, lat_trim, pedal_trim, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            ca = config["agent"]["lateralPID"]
        self.dt = config["dt"]
        self.gains = np.array([ca["Ky"], ca["Ky_int"], ca["Ky_dot"],
                               np.rad2deg(ca["Kphi"]), np.rad2deg(ca["Kphi"]), np.rad2deg(ca["Kp"]),
                               np.rad2deg(ca["Kpsi"]), np.rad2deg(ca["Kpsi_int"]), np.rad2deg(ca["Kr"])])
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
        phi_error = phi_req - phi
        psi_error = psi_req - psi

        lat = np.clip(self.lat_trim + (Kphi * phi_error + Kphi_int * self.phi_int + Kp * p)/100, 0, 1)
        ped = np.clip(self.pedal_trim + (Kpsi * psi_error + Kpsi_int * self.psi_int + Kr * r)/100, 0, 1)

        self.y_req_int += y_error * self.dt
        self.phi_int += phi_error * self.dt
        self.psi_int += psi_error * self.dt

        return lat, ped








