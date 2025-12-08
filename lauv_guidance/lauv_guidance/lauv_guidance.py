import rclpy
from rclpy.node import Node
import numpy as np
import math
import tf_transformations

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path


# --- Helper Function (from MSS) ---
def ssa(angle):
    """Wraps angle to [-pi, pi)"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class LOSGuidance:
    def __init__(self, lookahead_dist, acceptance_radius):
        self.delta = lookahead_dist
        self.radius = acceptance_radius

        self.path = []  # List of np.array([x, y, z])
        self.current_idx = 0
        self.active = False

        # State
        self.position = np.zeros(3)  # x, y, z
        self.heading = 0.0  # yaw

        # Outputs
        self.desired_heading = 0.0
        self.cross_track_error = 0.0
        self.target_position = np.zeros(3)

    def set_path(self, path_points):
        """Sets a new path and resets state."""
        if len(path_points) < 2:
            return False

        self.path = [np.array(p) for p in path_points]
        self.current_idx = 1  # Target is the 2nd point (index 1)
        self.active = True
        return True

    def update_state(self, x, y, z, yaw):
        self.position = np.array([x, y, z])
        self.heading = yaw

    def update_control_law(self):
        """
        Main logic function. Updates current waypoint index and calculates steering.
        Returns: True if mission is active, False if finished.
        """
        if not self.active or not self.path:
            return False

        # 1. Waypoint Switching Logic
        p_target = self.path[self.current_idx]
        dist_to_target = np.linalg.norm(p_target[0:2] - self.position[0:2])

        if dist_to_target < self.radius:
            if self.current_idx < len(self.path) - 1:
                self.current_idx += 1
            else:
                self.active = False
                return False  # Mission Finished

        # 2. LOS Calculations (MSS crosstrackWpt & LOSchi logic)
        p_prev = self.path[self.current_idx - 1]
        p_curr = self.path[self.current_idx]

        # Path Tangential Angle (alpha_k / pi_p)
        pi_p = math.atan2(p_curr[1] - p_prev[1], p_curr[0] - p_prev[0])

        # Cross-Track Error (e)
        dx = self.position[0] - p_prev[0]
        dy = self.position[1] - p_prev[1]

        # Rotate error into path frame
        self.cross_track_error = -dx * math.sin(pi_p) + dy * math.cos(pi_p)

        # LOS Law: chi_d = pi_p - atan(e / Delta)
        self.desired_heading = ssa(
            pi_p - math.atan(self.cross_track_error / self.delta)
        )

        # 3. Target Z (Depth)
        self.target_position[2] = p_curr[2]

        return True

    def get_surge_target(self, target_dist):
        """
        Calculates the target position for the position controller.
        """
        x_des = self.position[0] + target_dist * math.cos(self.desired_heading)
        y_des = self.position[1] + target_dist * math.sin(self.desired_heading)
        return x_des, y_des


# =============================================================================
# ROS NODE WRAPPER
# =============================================================================
class LOSGuidanceNode(Node):
    def __init__(self):
        super().__init__("los_guidance_node")
        self.get_logger().info("Initializing LOS Guidance Node...")

        # --- ROS Parameters ---
        self.declare_parameter("lookahead_distance", 5.0)
        self.declare_parameter("acceptance_radius", 2.0)
        self.declare_parameter("virtual_target_distance", 5.0)
        self.declare_parameter("default_depth", -10.0)

        # --- Core Logic Instance ---
        delta = self.get_parameter("lookahead_distance").value
        rad = self.get_parameter("acceptance_radius").value
        self.guidance = LOSGuidance(delta, rad)

        # --- Interfaces ---
        self.path_sub = self.create_subscription(
            Path, "/lauv/global_path", self.path_callback, 10
        )
        # TODO chage to odom_filtered when nav is implemented
        # self.odom_sub = self.create_subscription(
        #     Odometry, "lauv/odom_filtered", self.odom_callback, 10
        # )
        self.odom_sub = self.create_subscription(
            Odometry, "/lauv/odometry", self.odom_callback, 10
        )

        self.ref_pub = self.create_publisher(
            PoseStamped, "/lauv/ref_trajectory_filtered", 10
        )

        self.los_pub = self.create_publisher(Point, "/lauv/debug/los_point", 10)

        # Run loop at 10Hz
        self.timer = self.create_timer(0.1, self.control_loop)

    def path_callback(self, msg):
        path_list = []
        for p in msg.poses:
            path_list.append([p.pose.position.x, p.pose.position.y, p.pose.position.z])

        if self.guidance.set_path(path_list):
            self.get_logger().info(
                f"Path received with {len(path_list)} waypoints. Active."
            )
        else:
            self.get_logger().warn("Invalid path received (need >= 2 points).")

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.guidance.update_state(p.x, p.y, p.z, yaw)

    def control_loop(self):
        # Update tuning params dynamically (optional, but good practice)
        self.guidance.delta = self.get_parameter("lookahead_distance").value
        self.guidance.radius = self.get_parameter("acceptance_radius").value
        target = self.get_parameter("virtual_target_distance").value
        def_depth = self.get_parameter("default_depth").value

        # prinnt values
        self.get_logger().info(
            f"LOSGuidance Params - Lookahead: {self.guidance.delta}, Acceptance Radius: {self.guidance.radius}, Virtual Target Dist: {target}, Default Depth: {def_depth}",
            throttle_duration_sec=10.0,
        )

        # Run Guidance Law
        is_active = self.guidance.update_control_law()

        if is_active:
            # Calculate Target Position for Surge
            x_des, y_des = self.guidance.get_surge_target(target)

            # Handle Depth (if path has 0.0, use default)
            z_des = self.guidance.target_position[2]
            if z_des == 0.0:
                z_des = def_depth

            # Publish
            self.publish_command(x_des, y_des, z_des, self.guidance.desired_heading)
            self.publish_debug(x_des, y_des, z_des)

        elif self.guidance.path and not is_active:
            # Mission just finished or stopped
            # Hold current position
            self.get_logger().info(
                "Holding Position (Mission Complete/Inactive)",
                throttle_duration_sec=5.0,
            )
            self.publish_command(
                self.guidance.position[0],
                self.guidance.position[1],
                0.0,  # Surface or hold depth? Adjust as needed.
                self.guidance.heading,
            )

    def publish_command(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # msg.header.frame_id = "map"

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)

        # Orientation (Yaw only, Pitch handled by Depth controller)
        q = tf_transformations.quaternion_from_euler(0, 0, yaw)
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
    node = LOSGuidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
