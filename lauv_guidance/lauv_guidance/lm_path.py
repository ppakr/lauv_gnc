import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger


class LawnMowerPathNode(Node):
    def __init__(self):
        super().__init__("lawn_mower_path_node")
        self.get_logger().info("LawnMowerPath Node Initialized (Horizontal Mode)")

        self.waypoints = []

        # --- set up ROS ---
        self.path_pub = self.create_publisher(Path, "lauv/global_path", 10)

        self.srv = self.create_service(
            Trigger, "publish_lawn_mower_path", self.trigger_callback
        )

        # Create a timer to publish the path periodically (1 Hz)
        self.timer = self.create_timer(1.0, self.pub_global_path)

    def trigger_callback(self, request, response):
        """
        Service callback to publish the lawn mower path on demand.
        """
        self.pub_global_path()
        response.success = True
        response.message = "Lawn mower path published."
        return response

    def generate_lawn_mower(self, start_x, start_y, leg_length, spacing, num_legs):
        """
        Generates a HORIZONTAL lawn mower path (X-axis legs) with Dubins U-turns.
        """
        self.waypoints = []  # Reset waypoints

        # --- Generation Loop ---
        for i in range(num_legs):
            # Calculate Y for this leg (moves Up/North each leg)
            current_y = start_y + (i * spacing)

            # Determine Direction
            is_even = i % 2 == 0  # Even = Right/East, Odd = Left/West

            if is_even:
                # --- GOING RIGHT (West -> East) ---
                # 1. Start of leg
                if i == 0:
                    self.waypoints.append((start_x, current_y))

                # 2. Straight line to Right End
                self.waypoints.append((start_x + leg_length, current_y))

                # 3. U-Turn at Right End (Left Turn / CCW)
                if i < num_legs - 1:
                    center_x = start_x + leg_length
                    center_y = current_y + (spacing / 2.0)

                    # Arc: Starts at Bottom (-pi/2), goes CCW to Top (pi/2)
                    # Bulges to the East (Positive X)
                    self.add_arc(
                        center_x, center_y, -np.pi / 2, np.pi / 2, spacing / 2.0
                    )

            else:
                # --- GOING LEFT (East -> West) ---
                # Straight line to Left End
                self.waypoints.append((start_x, current_y))

                # U-Turn at Left End (Right Turn / CW)
                if i < num_legs - 1:
                    center_x = start_x
                    center_y = current_y + (spacing / 2.0)

                    # Arc: Starts at Bottom (3pi/2), goes CW to Top (pi/2)
                    # Bulges to the West (Negative X)
                    self.add_arc(
                        center_x, center_y, 3 * np.pi / 2, np.pi / 2, spacing / 2.0
                    )

        self.get_logger().info(f"Generated Path with {len(self.waypoints)} waypoints.")
        return self.waypoints

    # Helper to generate arc points
    def add_arc(
        self, center_x, center_y, theta_start, theta_end, radius, num_points=15
    ):
        # Generate angles.
        angles = np.linspace(theta_start, theta_end, num_points)

        for ang in angles:
            wx = center_x + radius * np.cos(ang)
            wy = center_y + radius * np.sin(ang)
            self.waypoints.append((wx, wy))

    def pub_global_path(self):
        """
        Converts internal waypoints to a ROS Path message and publishes it.
        """
        if not self.waypoints:
            self.get_logger().warn("No waypoints generated yet!")
            return

        msg = Path()
        # msg.header.frame_id = "map"  # Ensure this matches your localization/world frame
        msg.header.stamp = self.get_clock().now().to_msg()

        for wp in self.waypoints:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = -10.0  # Default depth

            # Orientation is not strictly needed for the path itself (LOS calculates it),
            # but we leave it as identity (0,0,0,1)
            pose.pose.orientation.w = 1.0

            msg.poses.append(pose)

        self.path_pub.publish(msg)
        self.get_logger().info(
            f"Published path with {len(msg.poses)} poses", throttle_duration_sec=5
        )

        # --- One-Shot Logic ---
        # Cancel the timer after the first execution so it doesn't loop
        if not self.timer.is_canceled():
            self.timer.cancel()
            self.get_logger().info(
                "Initialization timer cancelled (One-Shot complete)."
            )


def main(args=None):
    rclpy.init(args=args)

    # --- Configuration ---
    start_x = 0.0
    start_y = 0.0
    leg_length = 20.0  # Horizontal length
    spacing = 5.0  # Vertical step
    num_legs = 2

    # --- Node Setup ---
    node = LawnMowerPathNode()

    # Generate the points
    node.generate_lawn_mower(start_x, start_y, leg_length, spacing, num_legs)

    # Keep the node alive to publish the path periodically
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
