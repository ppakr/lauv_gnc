import rclpy
from rclpy.node import Node
import numpy as np
import math
import tf_transformations

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64


class LOSGuidance(Node):
    def __init__(self):
        super().__init__("los_guidance_node")
        # TODO: Parameters
        # TODO: ROS interfaces

    # TODO path callback
    # TODO odom callback
    # TODO control loop
    # TODO publish command


def main(args=None):
    rclpy.init(args=args)
    node = LOSGuidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
