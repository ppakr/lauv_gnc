import pytest
import numpy as np
import rclpy
from lauv_control.lauv_control import LAUVControl


# --- Fixture to handle ROS 2 init/shutdown ---
@pytest.fixture(scope="function")
def ros_node_setup():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_depth_control_decoupled(ros_node_setup):
    """
    Test 1: Verify Depth Error directly drives Heave Velocity (w).
    """
    node = LAUVControl()

    # Configure Depth PID: Kp=1.0
    node.pid_z.k_p = 1.0
    node.pid_z.k_i = 0.0
    node.pid_z.k_d = 0.0

    # Increase saturation limit for test
    node.pid_z.sat = 100.0

    # Scenario:
    # Robot is at Depth=0. Desired Depth=10.
    node.eta_actual = np.zeros((6, 1))
    node.eta_desired = np.zeros((6, 1))
    node.eta_desired[2, 0] = 10.0

    node.prev_t = 0.0
    node.time = 1.0

    # Run Logic
    node.position_control()

    # Verify Logic
    # Error = 10.0 - 0.0 = 10.0
    # Expected Heave Velocity (w) = 10.0 * 1.0 = 10.0
    actual_w_cmd = node.nu_desired[2, 0]

    assert actual_w_cmd == pytest.approx(10.0, abs=0.01)


def test_pitch_control_independent(ros_node_setup):
    """
    Test 2: Verify Pitch Error directly drives Pitch Rate (q).
    """
    node = LAUVControl()

    # Configure Pitch PID: Kp=2.0
    node.pid_theta.k_p = 2.0
    node.pid_theta.k_i = 0.0
    node.pid_theta.k_d = 0.0

    # FIX: Increase saturation limit so output isn't clamped at 0.5
    node.pid_theta.sat = 10.0

    # Scenario: Level (0), want to Pitch Up (0.5 rad)
    node.eta_actual = np.zeros((6, 1))
    node.eta_desired = np.zeros((6, 1))
    node.eta_desired[4, 0] = 0.5

    node.prev_t = 0.0
    node.time = 1.0

    node.position_control()

    # Verify Logic
    # Error = 0.5 - 0.0 = 0.5
    # Expected Pitch Rate (q) = 0.5 * 2.0 = 1.0
    actual_q_cmd = node.nu_desired[4, 0]

    assert actual_q_cmd == pytest.approx(1.0, abs=0.01)


def test_yaw_control(ros_node_setup):
    """
    Test 3: Verify Yaw Error drives Yaw Rate (r).
    """
    node = LAUVControl()

    # Set Heading PID Kp=2.0
    node.pid_psi.k_p = 2.0

    # FIX: Increase saturation limit
    node.pid_psi.sat = 10.0

    # Scenario: Face North (0), Want West (PI/2)
    node.eta_actual = np.zeros((6, 1))
    node.eta_desired = np.zeros((6, 1))
    node.eta_desired[5, 0] = np.pi / 2

    node.prev_t = 0.0
    node.time = 1.0

    node.position_control()

    # Error = pi/2
    # Output (r) = 2.0 * (pi/2) = pi (approx 3.14)
    expected_r = np.pi

    actual_r_cmd = node.nu_desired[5, 0]

    assert actual_r_cmd == pytest.approx(expected_r, abs=0.01)


def test_full_dof_mapping(ros_node_setup):
    """
    Test 4: Sanity check that all 6 DOFs produce non-zero outputs
    when errors exist.
    """
    node = LAUVControl()

    # Set all PIDs to Kp=1.0 and Saturation=10.0
    pids = [
        node.pid_x,
        node.pid_y,
        node.pid_z,
        node.pid_phi,
        node.pid_theta,
        node.pid_psi,
    ]
    for pid in pids:
        pid.k_p = 1.0
        pid.k_i = 0.0
        pid.k_d = 0.0
        pid.sat = 10.0  # Unclamp for testing

    # Create error of 1.0 in every axis
    node.eta_desired = np.ones((6, 1))
    node.eta_actual = np.zeros((6, 1))

    node.prev_t = 0.0
    node.time = 1.0

    node.position_control()

    # Check that all velocity commands are approx 1.0
    for i in range(6):
        assert node.nu_desired[i, 0] == pytest.approx(1.0, abs=0.01), f"DOF {i} failed"
