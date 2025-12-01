import pytest
import numpy as np
from lauv_control.pid_controller import PIDController


class TestPIDController:
    def test_initialization(self):
        pid = PIDController(k_p=1.0, k_i=0.1, k_d=0.01, sat=5.0)
        assert pid.k_p == 1.0
        assert pid.sat == 5.0
        assert pid.type == "linear"

    def test_linear_error(self):
        pid = PIDController(type="linear")
        # Desired=10, Actual=5 -> Error=5
        err, _, _ = pid.calculate_error(desired=10.0, actual=5.0, t=1.0)
        assert err == 5.0

    def test_angular_error_wrapping(self):
        pid = PIDController(type="angular")

        # Case 1: Simple difference
        # Desired=0.1, Actual=0.0 -> Error=0.1
        err, _, _ = pid.calculate_error(0.1, 0.0, 1.0)
        assert err == pytest.approx(0.1)

        # Case 2: Wrapping across Pi
        # Desired = pi - 0.1 (approx 3.04)
        # Actual  = -pi + 0.1 (approx -3.04)
        # Expected error is roughly -0.2 (shortest path), not +6.0
        target = np.pi - 0.1
        current = -np.pi + 0.1
        err, _, _ = pid.calculate_error(target, current, 2.0)

        # The error should be negative (go the other way)
        assert err == pytest.approx(-0.2)

    def test_saturation(self):
        # Setup high gains to force saturation
        pid = PIDController(k_p=1000.0, sat=10.0)

        # Calc error
        err, int_err, diff_err = pid.calculate_error(100.0, 0.0, 1.0)

        # Calc Output
        output = pid.calculate_pid(err, int_err, diff_err)

        # Should be capped at 10.0
        assert output == 10.0

        # Test negative saturation
        err, int_err, diff_err = pid.calculate_error(-100.0, 0.0, 2.0)
        output = pid.calculate_pid(err, int_err, diff_err)
        assert output == -10.0
