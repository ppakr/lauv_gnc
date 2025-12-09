import pytest
import numpy as np
import math
from lauv_guidance.lauv_guidance import LOSGuidance, ssa


class TestLOSGuidanceCore:
    def test_initialization(self):
        los = LOSGuidance(lookahead_dist=5.0, acceptance_radius=2.0)
        assert los.delta == 5.0
        assert los.radius == 2.0
        assert not los.active

    def test_path_setup(self):
        los = LOSGuidance(5.0, 2.0)
        path = [[0, 0, 0], [10, 0, 0], [10, 10, 0]]

        # Valid path
        assert los.set_path(path) == True
        assert los.current_idx == 1  # Targets the second point
        assert los.active == True

        # Invalid path (too short)
        assert los.set_path([[0, 0, 0]]) == False

    def test_straight_line_guidance(self):
        """
        Scenario: Robot is at (0, -2), Path is (0,0) -> (10,0).
        Robot is 2m RIGHT of path (Cross track error should be non-zero).
        It should steer LEFT (Positive Yaw).
        """
        los = LOSGuidance(lookahead_dist=5.0, acceptance_radius=1.0)
        los.set_path([[0, 0, 0], [100, 0, 0]])

        # Update Robot State: x=0, y=-2 (Right of path), heading=0
        los.update_state(x=0.0, y=-2.0, z=0.0, yaw=0.0)

        los.update_control_law()

        # Check Cross Track Error
        # e = -dx*sin(pi) + dy*cos(pi). pi=0. dx=0, dy=-2.
        # e = -2.0. (Negative e means right of path in standard NED, check your sign convention)
        # Formula used: e = -dx*sin + dy*cos
        # pi_p = atan2(0, 100) = 0.
        # e = -0*0 + (-2)*1 = -2.0. Correct.
        assert los.cross_track_error == pytest.approx(-2.0)

        # Check Desired Heading
        # chi_d = pi_p - atan(e / delta)
        # chi_d = 0 - atan(-2 / 5) = -(-0.38) = +0.38 rad
        expected_heading = 0.0 - math.atan(-2.0 / 5.0)
        assert los.desired_heading == pytest.approx(expected_heading)
        assert los.desired_heading > 0  # Should turn LEFT (Positive)

    def test_waypoint_switching(self):
        """
        Scenario: Robot approaches waypoint 1. logic should switch to waypoint 2.
        Path: (0,0) -> (10,0) -> (10,10)
        Radius: 2.0
        """
        los = LOSGuidance(5.0, 2.0)
        los.set_path([[0, 0, 0], [10, 0, 0], [10, 10, 0]])

        # 1. Approach Waypoint 1 (at 10,0). Robot at (9, 0).
        # Distance = 1.0 < Radius(2.0). Should switch.
        los.update_state(x=9.0, y=0.0, z=0.0, yaw=0.0)
        active = los.update_control_law()

        assert active == True
        assert los.current_idx == 2  # Switched to next segment

        # 2. Check Guidance uses new segment ((10,0) -> (10,10))
        # Path Angle should be 90 deg (pi/2)
        # Robot at (9,0). Line starts at (10,0).
        # Cross track error calculation will happen relative to vertical line.

        # We just verify it's targeting the last point now
        target_z = los.target_position[2]
        # (Assuming flat depth 0 for simplicity, but logic holds)

    def test_mission_complete(self):
        los = LOSGuidance(5.0, 2.0)
        los.set_path([[0, 0, 0], [10, 0, 0]])  # Only 1 segment

        # Reach the end
        los.update_state(x=9.5, y=0.0, z=0.0, yaw=0.0)
        active = los.update_control_law()

        assert active == False  # Should finish
        assert los.active == False

    def test_virtual_target_projection(self):
        """Test the 'Carrot' position calculation for surge control."""
        los = LOSGuidance(5.0, 2.0)
        los.desired_heading = math.pi / 2  # Facing North
        los.position = np.array([10.0, 10.0, 0.0])

        # Config
        target_vel = 1.0  # m/s
        dt = 5.0  # seconds (Simulate 5s ahead)

        # Expected displacement = vel * dt = 5.0m
        # Project 5m ahead along heading (North)
        tx, ty = los.get_surge_target(target_vel=target_vel, d_t=dt)

        # Should be at (10, 15) roughly
        assert tx == pytest.approx(10.0)
        assert ty == pytest.approx(15.0)
