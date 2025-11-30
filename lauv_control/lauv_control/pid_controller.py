import numpy as np
from copy import deepcopy


class PIDController:
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, sat=10.0, type='linear'):
        # variables declaration
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.sat = sat

        self.PID = 0.0

        self.err = 0.0
        self.prev_err = 0.0
        self.diff_err = 0.0
        self.int_err = 0.0

        self.t = 0.0
        self.prev_t = -1.0

        self.type = type

    def calculate_error(self, desired, actual, t):

        # calculating error

        self.err = desired - actual

        # check the type of input (linear or angular)

        # if linear, do nothing

        if self.type == 'linear':
            pass

        # if angular, bound the output

        elif self.type == 'angular':
            if self.err > np.pi:
                self.err = self.err - (2.0 * np.pi)
            elif self.err < -np.pi:
                self.err = self.err + (2.0 * np.pi)

        # calculating dt

        self.t = t
        dt = self.t - self.prev_t

        if (self.prev_t == -1.0):  # first time

            self.diff_err = 0.0
            self.int_err = 0.0

        elif dt > 0.0:

            # calculate derivative error

            self.diff_err = (self.err - self.prev_err) / dt

            # calucalte integral error

            self.int_err = self.int_err + \
                ((self.err + self.prev_err) * dt / 2.0)

        self.prev_t = deepcopy(self.t)

        return self.err, self.int_err, self.diff_err

    def calculate_pid(self, err, int_err, diff_err):

        # calculate PID

        self.PID = (self.k_p * err) + (self.k_i *
                                       int_err) + (self.k_d * diff_err)

        self.prev_err = deepcopy(self.err)

        return self.PID

    def reconfig_param(self, k_p, k_i, k_d):
        # update the variables
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        # reset integral
        self.int_err = 0.0
    
    def reset_control(self):
        self.PID = 0.0
        self.err = 0.0
        self.prev_err = 0.0
        self.diff_err = 0.0
        self.int_err = 0.0
