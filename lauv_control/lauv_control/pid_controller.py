import numpy as np
from copy import deepcopy


class PIDController:
    """
    A simple Proportional-Integral-Derivative (PID) controller.

    Supports both linear and angular error calculations. Angular mode
    handles angle wrapping between -pi and pi.
    """

    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, sat=10.0, type="linear"):
        """
        Initializes the PID controller with gains and constraints.

        Args:
            k_p (float): Proportional gain.
            k_i (float): Integral gain.
            k_d (float): Derivative gain.
            sat (float): Saturation limit for output (currently unused in logic).
            type (str): Type of control, either "linear" or "angular".
        """
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
        """
        Computes the proportional, integral, and derivative errors.

        Args:
            desired (float): The target value.
            actual (float): The current measured value.
            t (float): Current timestamp.

        Returns:
            tuple: (error, integral_error, derivative_error)
        """

        # calculating error
        self.err = desired - actual

        # check the type of input (linear or angular)

        # if linear, do nothing
        if self.type == "linear":
            pass

        # if angular, bound the output
        elif self.type == "angular":
            if self.err > np.pi:
                self.err = self.err - (2.0 * np.pi)
            elif self.err < -np.pi:
                self.err = self.err + (2.0 * np.pi)

        # calculating dt
        self.t = t
        dt = self.t - self.prev_t

        if self.prev_t == -1.0:  # first time

            self.diff_err = 0.0
            self.int_err = 0.0

        elif dt > 0.0:

            # calculate derivative error
            self.diff_err = (self.err - self.prev_err) / dt

            # calculate integral error
            self.int_err = self.int_err + ((self.err + self.prev_err) * dt / 2.0)

            # integral windup guard
            if self.int_err > self.sat:
                self.int_err = self.sat
            elif self.int_err < -self.sat:
                self.int_err = -self.sat

        self.prev_t = deepcopy(self.t)

        return self.err, self.int_err, self.diff_err

    def calculate_pid(self, err, int_err, diff_err):
        """
        Calculates the final PID control output based on error terms.

        Args:
            err (float): Proportional error.
            int_err (float): Accumulated integral error.
            diff_err (float): Derivative error (rate of change).

        Returns:
            float: The calculated control output.
        """

        # calculate PID
        self.PID = (self.k_p * err) + (self.k_i * int_err) + (self.k_d * diff_err)

        # bound output
        if self.PID > self.sat:
            self.PID = self.sat
        elif self.PID < -self.sat:
            self.PID = -self.sat

        self.prev_err = deepcopy(self.err)

        return self.PID

    def reconfig_param(self, k_p, k_i, k_d):
        """
        Updates the PID gains dynamically and resets the integral error.

        Args:
            k_p (float): New proportional gain.
            k_i (float): New integral gain.
            k_d (float): New derivative gain.
        """
        # update the variables
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        # reset integral
        self.int_err = 0.0

    def reset_control(self):
        """
        Resets all error terms and control outputs to zero.
        """
        self.PID = 0.0
        self.err = 0.0
        self.prev_err = 0.0
        self.diff_err = 0.0
        self.int_err = 0.0
