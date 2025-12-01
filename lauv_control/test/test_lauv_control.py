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


def test_depth_to_pitch_coupling(ros_node_setup):
    # 1. Instantiate the controller (Safe now that rclpy is initialized)
    node = LAUVControl()

    # 2. Configure PID gains for simple math in test
    # Depth PID: Kp=1.0 implies Output = Error * 1.0
    node.pid_z.k_p = 1.0
    node.pid_z.k_i = 0.0
    node.pid_z.k_d = 0.0

    # Pitch PID: Kp=1.0
    node.pid_theta.k_p = 1.0

    # 3. Setup Scenario:
    # Robot is at Depth=0, Desired Depth=10.
    node.eta_actual = np.zeros((6, 1))
    node.eta_desired = np.zeros((6, 1))
    node.eta_desired[2, 0] = 10.0  # Target Depth = 10m

    # 4. Manually set time to ensure dt is calculated correctly
    node.prev_t = 0.0
    node.time = 1.0

    # 5. Run Logic
    node.position_control()

    # 6. Verify Logic
    # Depth Error = 10.0 -> PID_z Output = 10.0 (Pitch Offset)
    # Desired Pitch (0) - Pitch Offset (10.0) = -10.0 rad
    # Clamped to -45 deg (-0.785 rad)
    expected_pitch_command = -0.785398

    # Check Pitch Rate Output (q)
    # Error = Desired(-0.785) - Actual(0) = -0.785
    # PID_theta Output = -0.785 * 1.0 = -0.785

    actual_q_cmd = node.nu_desired[4, 0]

    # Clean up node resource
    node.destroy_node()

    # Assert that we are commanding a NEGATIVE pitch rate (Nose down to dive)
    assert actual_q_cmd < 0.0
    assert actual_q_cmd == pytest.approx(expected_pitch_command, abs=0.01)


def test_yaw_control(ros_node_setup):
    node = LAUVControl()

    # Set Heading PID Kp=2.0
    node.pid_psi.k_p = 2.0

    # Scenario: Face North (0), Want West (PI/2)
    node.eta_actual = np.zeros((6, 1))
    node.eta_desired = np.zeros((6, 1))
    node.eta_desired[5, 0] = np.pi / 2

    node.prev_t = 0.0
    node.time = 1.0

    node.position_control()

    node.destroy_node()

    # Error = pi/2
    # Output (r) = 2.0 * (pi/2) = pi
    expected_r = np.pi

    assert node.nu_desired[5, 0] == pytest.approx(expected_r, abs=0.01)
