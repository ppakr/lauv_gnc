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
        self.get_logger().info("Initializing LOS Guidance Node...")

        # --- Parameters ---
        # Lookahead Distance (Delta): Tunable.
        # Larger = smoother/lazy steering. Smaller = aggressive/oscillating.
        self.declare_parameter("lookahead_distance", 5.0)
        self.declare_parameter(
            "acceptance_radius", 2.0
        )  # Switch waypoint when closer than this
        self.declare_parameter(
            "surge_carrot_dist", 5.0
        )  # How far ahead to project target for surge control
        self.declare_parameter(
            "default_depth", 1.0
        )  # Target depth if not specified in path

        # --- State ---
        self.active = False
        self.current_pose = None  # [x, y, z, roll, pitch, yaw]
        self.path = []  # List of [x, y, z] points
        self.current_idx = 0  # Index of the "To" waypoint

        # --- ROS Interfaces ---
        # User sends path here (e.g., from a script or Rviz)
        self.path_sub = self.create_subscription(
            Path, "lauv/global_path", self.path_callback, 10
        )

        # Robot State (Odometry)
        self.odom_sub = self.create_subscription(
            Odometry, "lauv/odom_filtered", self.odom_callback, 10
        )

        # Output: Command to Controller
        self.ref_pub = self.create_publisher(
            PoseStamped, "lauv/ref_trajectory_filtered", 10
        )

        # Debug: Publish Lookahead Point (Visualize in Rviz)
        self.los_pub = self.create_publisher(Point, "lauv/debug/los_point", 10)

        self.timer = self.create_timer(0.1, self.control_loop)  # 10Hz

    # TODO path callback
    def path_callback(self, msg):
        """API: User commands a new path."""
        if len(msg.poses) < 2:
            self.get_logger().warn("Path must have at least 2 waypoints!")
            return

        self.path = []
        for p in msg.poses:
            self.path.append([p.pose.position.x, p.pose.position.y, p.pose.position.z])

        self.current_idx = 1  # Start moving towards the second point (0 -> 1)
        self.active = True
        self.get_logger().info(
            f"Received Path with {len(self.path)} waypoints. Starting Mission."
        )

    # TODO odom callback
    def odom_callback(self, msg):
        """Update robot state."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        # Convert Quat to Euler
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        self.current_pose = np.array([p.x, p.y, p.z, roll, pitch, yaw])

    # TODO control loop
    def control_loop(self):
        pass

    # TODO publish command
    def publish_command(self, x, y, z, yaw, pitch):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  # or "world" / "odom"

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        # Orientation (Yaw + Pitch)
        # Note: LauvController usually only needs Yaw for heading, Pitch for depth
        q = tf_transformations.quaternion_from_euler(0, pitch, yaw)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        self.ref_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LOSGuidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
