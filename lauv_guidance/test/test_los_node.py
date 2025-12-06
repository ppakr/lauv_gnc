import pytest
import rclpy
from lauv_guidance.lauv_guidance import LOSGuidanceNode
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


@pytest.fixture(scope="function")
def ros_node_setup():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_node_init(ros_node_setup):
    node = LOSGuidanceNode()
    assert node.guidance.delta == 5.0  # Default param check
    node.destroy_node()


def test_path_callback_ros(ros_node_setup):
    node = LOSGuidanceNode()

    # Create a dummy path message
    msg = Path()
    p1 = PoseStamped()
    p1.pose.position.x = 0.0
    p2 = PoseStamped()
    p2.pose.position.x = 10.0
    msg.poses = [p1, p2]

    # Call callback manually (simulating ROS msg)
    node.path_callback(msg)

    # Check if core logic received it
    assert node.guidance.active == True
    assert len(node.guidance.path) == 2
    node.destroy_node()
