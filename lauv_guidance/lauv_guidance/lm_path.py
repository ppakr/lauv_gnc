import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger


class LawnMowerPathNode(Node):
    def __init__(self):
        super().__init__("lawn_mower_path_node")
        self.get_logger().info("LawnMowerPath Node Initialized")

        self.waypoints = []

        # --- set up ROS ---
        # Note: Updated topic to match your Guidance Node's subscriber
        self.path_pub = self.create_publisher(Path, "lauv/global_path", 10)

        self.srv = self.create_service(
            Trigger, "publish_lawn_mower_path", self.trigger_callback
        )

        # Create a timer to publish the path periodically (1 Hz)
        # This ensures the guidance node receives it even if it starts late.
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
        Generates a simple lawn mower path with Dubins U-turns.
        """
        self.waypoints = []  # Reset waypoints

        # --- Generation Loop ---
        for i in range(num_legs):
            # Calculate X for this leg
            current_x = start_x + (i * spacing)

            # Determine Direction
            is_even = i % 2 == 0  # Even = UP, Odd = DOWN

            if is_even:
                # --- GOING UP (South -> North) ---
                # Start of leg
                if i == 0:
                    self.waypoints.append((current_x, start_y))

                # Straight line to Top
                self.waypoints.append((current_x, start_y + leg_length))

                # U-Turn at Top (Right Turn / Clockwise)
                if i < num_legs - 1:
                    center_x = current_x + (spacing / 2.0)
                    center_y = start_y + leg_length

                    # Arc: Starts at West point (pi), goes Clockwise to East point (0)
                    # Curve is ABOVE the line (sin is positive)
                    self.add_arc(center_x, center_y, np.pi, 0, spacing / 2.0)
            else:
                # --- GOING DOWN (North -> South) ---
                # Straight line to Bottom
                # (We are already at the top from the previous turn)
                self.waypoints.append((current_x, start_y))

                # U-Turn at Bottom (Left Turn / Counter-Clockwise)
                if i < num_legs - 1:
                    center_x = current_x + (spacing / 2.0)
                    center_y = start_y

                    # Arc: Starts at West point (pi), goes Counter-Clockwise to East point (2pi)
                    # Curve is BELOW the line (sin is negative)
                    self.add_arc(center_x, center_y, np.pi, 2 * np.pi, spacing / 2.0)

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

        # if service triggered

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

            self.get_logger().info(f"Waypoint: x={wp[0]}, y={wp[1]}")

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
    leg_length = 50.0
    spacing = 5.0
    num_legs = 6
    # Note: radius should ideally be spacing / 2 for perfect U-turns
    radius = spacing / 2.0

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
