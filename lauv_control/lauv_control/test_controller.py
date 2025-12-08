import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf_transformations
import numpy as np
import sys


class ControllerTester(Node):
    def __init__(self):
        super().__init__("controller_tester")

        # Publisher (Sends commands to Controller)
        self.cmd_pub = self.create_publisher(
            PoseStamped, "/lauv/ref_trajectory_filtered", 10
        )

        # Subscriber (To get current pos for relative commands)
        self.odom_sub = self.create_subscription(
            Odometry, "/lauv/odometry", self.odom_callback, 10
        )

        self.current_pose = None
        self.timer = self.create_timer(0.1, self.loop)

        # Get Test Mode from args
        # 1 = Surge, 2 = Dive, 3 = Yaw
        self.mode = int(sys.argv[1]) if len(sys.argv) > 1 else 1
        self.get_logger().info(f"Starting Test Mode: {self.mode}")

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def loop(self):
        if self.current_pose is None:
            self.get_logger().warn("Waiting for Odom...", throttle_duration_sec=2.0)
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # Get current state
        curr_x = self.current_pose.position.x
        curr_y = self.current_pose.position.y
        curr_z = self.current_pose.position.z

        # Extract Yaw
        q = self.current_pose.orientation
        _, _, curr_yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # --- TEST 1: SURGE ONLY ---
        # Strategy: Place target 50m ahead aligned with current heading
        if self.mode == 1:
            target_dist = 50.0
            msg.pose.position.x = curr_x + target_dist * np.cos(curr_yaw)
            msg.pose.position.y = curr_y + target_dist * np.sin(curr_yaw)
            msg.pose.position.z = curr_z  # Maintain depth

            # Maintain Heading
            q_new = tf_transformations.quaternion_from_euler(0, 0, curr_yaw)
            msg.pose.orientation.x = q_new[0]
            msg.pose.orientation.y = q_new[1]
            msg.pose.orientation.z = q_new[2]
            msg.pose.orientation.w = q_new[3]

        # --- TEST 2: SURGE + HEAVE (DIVE) ---
        # Strategy: 50m ahead, but Target Z is DEEPER (Negative in ENU?)
        # NOTE: In ENU, Z+ is Up. To dive, we typically want lower Z.
        elif self.mode == 2:
            target_dist = 50.0
            msg.pose.position.x = curr_x + target_dist * np.cos(curr_yaw)
            msg.pose.position.y = curr_y + target_dist * np.sin(curr_yaw)

            # COMMAND DIVE: Set target 5 meters below current
            # Check your frame! If Odom says Z=0 at surface, set Z=-5.0.
            msg.pose.position.z = -5.0

            # Maintain Heading
            q_new = tf_transformations.quaternion_from_euler(0, 0, curr_yaw)
            msg.pose.orientation.x = q_new[0]
            msg.pose.orientation.y = q_new[1]
            msg.pose.orientation.z = q_new[2]
            msg.pose.orientation.w = q_new[3]

        # --- TEST 3: SURGE + YAW (TURN) ---
        # Strategy: 50m ahead, but offset to the Left to force a turn
        elif self.mode == 3:
            target_dist = 50.0
            # Target is 90 degrees to the LEFT of current
            target_yaw = curr_yaw + 1.57  # 90 deg turn

            # Place point in that direction
            msg.pose.position.x = curr_x + target_dist * np.cos(target_yaw)
            msg.pose.position.y = curr_y + target_dist * np.sin(target_yaw)
            msg.pose.position.z = curr_z

            # Explicitly command the new Heading
            q_new = tf_transformations.quaternion_from_euler(0, 0, target_yaw)
            msg.pose.orientation.x = q_new[0]
            msg.pose.orientation.y = q_new[1]
            msg.pose.orientation.z = q_new[2]
            msg.pose.orientation.w = q_new[3]

        self.cmd_pub.publish(msg)
        self.get_logger().info(
            f"Pub Mode {self.mode}: Target Z={msg.pose.position.z:.2f}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = ControllerTester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
