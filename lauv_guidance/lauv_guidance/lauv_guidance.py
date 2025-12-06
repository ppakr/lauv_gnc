import rclpy
from rclpy.node import Node
import numpy as np
import math
import tf_transformations

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64


def ssa(angle):
    """
    Wraps angle to [-pi, pi)
    Based on MSS toolbox: LIBRARY/kinematics/ssa.m
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


class LOSGuidance(Node):
    def __init__(self):
        super().__init__("los_guidance_node")
        self.get_logger().info("Initializing LOS Guidance Node (MSS GNC Compatible)...")

        # --- Parameters ---
        # Lookahead Distance (Delta): Tunable.
        self.declare_parameter("lookahead_distance", 5.0)
        self.declare_parameter("acceptance_radius", 2.0)
        self.declare_parameter("surge_carrot_dist", 5.0)
        self.declare_parameter("default_depth", 1.0)

        # --- State ---
        self.active = False
        self.current_pose = None  # [x, y, z, roll, pitch, yaw]
        self.path = []  # List of [x, y, z] points
        self.current_idx = 0  # Index of the "To" waypoint

        # --- ROS Interfaces ---
        self.path_sub = self.create_subscription(
            Path, "gnc/global_path", self.path_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, "gnc/odom_filtered", self.odom_callback, 10
        )

        self.ref_pub = self.create_publisher(
            PoseStamped, "gnc/ref_trajectory_filtered", 10
        )

        self.los_pub = self.create_publisher(Point, "gnc/debug/los_point", 10)

        self.timer = self.create_timer(0.1, self.control_loop)  # 10Hz

    def crosstrackWpt(self, xk, yk, xk1, yk1, x, y):
        """
        Calculates cross-track error (e) and along-track distance (s).
        Based on MSS GNC/crosstrackWpt.m formulation.

        Args:
            xk, yk: Previous waypoint coordinates
            xk1, yk1: Next waypoint coordinates
            x, y: Current vehicle position

        Returns:
            s: Along-track distance
            e: Cross-track error
            pi_p: Path angle (alpha_k)
        """
        # Path angle (alpha_k)
        pi_p = math.atan2(yk1 - yk, xk1 - xk)

        # Relative position in path-aligned frame
        dx = x - xk
        dy = y - yk

        # Rotate to path frame
        # s = along-track, e = cross-track
        s = dx * math.cos(pi_p) + dy * math.sin(pi_p)
        e = -dx * math.sin(pi_p) + dy * math.cos(pi_p)

        return s, e, pi_p

    def LOSchi(self, x, y, xk, yk, xk1, yk1, Delta):
        """
        Line-of-Sight guidance law.
        Based on MSS GNC/LOSchi.m formulation.

        Returns:
            chi_d: Desired course angle (rad)
        """
        # Calculate errors
        _, e, pi_p = self.crosstrackWpt(xk, yk, xk1, yk1, x, y)

        # LOS Control Law: chi_d = alpha_k - atan(e / Delta)
        # Note: If e > 0 (left of path), we must steer right (negative correction).
        chi_d = pi_p - math.atan(e / Delta)

        return chi_d

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

    def odom_callback(self, msg):
        """Update robot state."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        # Convert Quat to Euler
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        self.current_pose = np.array([p.x, p.y, p.z, roll, pitch, yaw])

    def control_loop(self):
        if not self.active or self.current_pose is None or not self.path:
            return

        # Get Parameters
        delta = self.get_parameter("lookahead_distance").value
        radius = self.get_parameter("acceptance_radius").value
        carrot_dist = self.get_parameter("surge_carrot_dist").value

        # Get Waypoints (Previous -> Target)
        p_prev = np.array(self.path[self.current_idx - 1])
        p_target = np.array(self.path[self.current_idx])

        # Check Waypoint Switching (Circle of Acceptance)
        dist_to_target = np.linalg.norm(p_target[0:2] - self.current_pose[0:2])

        if dist_to_target < radius:
            self.get_logger().info(f"Reached Waypoint {self.current_idx}")
            if self.current_idx < len(self.path) - 1:
                self.current_idx += 1
                # Update line segment immediately
                p_prev = np.array(self.path[self.current_idx - 1])
                p_target = np.array(self.path[self.current_idx])
            else:
                self.get_logger().info("Mission Complete!")
                self.active = False
                # Stop: Publish current pose as target
                self.publish_command(
                    self.current_pose[0], self.current_pose[1], 0.0, 0.0, 0.0
                )
                return

        # Calculate Desired Heading (LOS)
        # Using MSS Logic
        chi_d = self.LOSchi(
            self.current_pose[0],
            self.current_pose[1],
            p_prev[0],
            p_prev[1],
            p_target[0],
            p_target[1],
            delta,
        )

        # Ensure angle is wrapped to [-pi, pi]
        psi_des = ssa(chi_d)

        # Project Ghost Target ("Carrot") for Position Controller
        # The controller needs a position (x,y) to drive the surge error.
        # We project a point `carrot_dist` meters ahead along the desired heading.
        x_des = self.current_pose[0] + carrot_dist * math.cos(psi_des)
        y_des = self.current_pose[1] + carrot_dist * math.sin(psi_des)

        # Depth Control
        z_des = p_target[2]
        if z_des == 0.0:
            z_des = self.get_parameter("default_depth").value

        # 6. Publish Command
        # Pitch is 0.0 because Depth controller handles pitch separately
        self.publish_command(x_des, y_des, z_des, psi_des, 0.0)

        # Debug
        self.publish_debug(x_des, y_des, z_des)

    def publish_command(self, x, y, z, yaw, pitch):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # msg.header.frame_id = "map"

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)

        # Orientation (Yaw + Pitch)
        q = tf_transformations.quaternion_from_euler(0, pitch, yaw)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        self.ref_pub.publish(msg)

    def publish_debug(self, x, y, z):
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        self.los_pub.publish(p)


def main(args=None):
    rclpy.init(args=args)
    node = LOSGuidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
