import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
import numpy as np
import math
from copy import deepcopy

from geometry_msgs.msg import TwistStamped, WrenchStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool

from lauv_control.pid_controller import PIDController


class LAUVControl(Node):
    def __init__(self):
        super().__init__("lauv_controller")
        self.get_logger().info("Initializing LAUV Controller...")

        # --- State Variables ---
        self.eta_desired = np.zeros((6, 1))  # [x, y, z, phi, theta, psi]
        self.eta_actual = np.zeros((6, 1))
        self.nu_desired = np.zeros((6, 1))  # [u, v, w, p, q, r]
        self.nu_actual = np.zeros((6, 1))
        self.acc = np.zeros((6, 1))  # Desired accelerations

        self.ref_trajectory_received = False
        self.odom_received = False
        self.is_control_on = True

        self.desired_pitch_limit = 0.26

        # --- PID set up ---
        self.config = {}

        # Position/Orientation Loop Gains
        self._declare_pid_params("x", 10.0, 1.0, 0.0)  # Surge Speed
        self._declare_pid_params("z", 0.5, 0.0, 0.0) #Depth Control
        self._declare_pid_params("phi", 0.05, 0.0, 0.0)  # Roll Angle (New)
        self._declare_pid_params("theta", 3.0, 0.0, 1.0)  # Pitch Angle
        self._declare_pid_params("psi", 3.0, 0.0, 0.0)  # Heading Angle

        # Callback for dynamic reconfigure
        self.add_on_set_parameters_callback(self.callback_params)

        # --- PID Instances (Full 6-DOF) ---
        # Outer Loop (Position -> Velocity)
        self.pid_x = PIDController(type="linear")
        self.pid_z = PIDController(type="linear", sat=10.0)
        self.pid_phi = PIDController(type="angular", sat=0.5)  # New
        self.pid_theta = PIDController(type="angular", sat=0.5)
        self.pid_psi = PIDController(type="angular", sat=0.5)

        # Initialize PID values from params
        self.update_control_param()

        self.get_logger().info("LAUV Controller Parameters Initialized.")

        # --- ROS 2 interfaces ---
        self.tau = WrenchStamped()
        self.nu_msg = TwistStamped()

        self.odom_sub = self.create_subscription(
            Odometry, "/lauv/odometry", self.odom_callback, 10
        )

        self.cmd_odom_sub = self.create_subscription(
            Odometry, "/lauv/ref_trajectory_filtered", self.cmd_odom_callback, 10
        )

        self.torque_pub = self.create_publisher(
            WrenchStamped, "/lauv/wrench_command", 10
        )

        self.ref_vel_pub = self.create_publisher(TwistStamped, "/lauv/cmd_vel", 10)

        # Timing
        self.control_rate = 20.0  # Hz
        self.control_period = 1.0 / self.control_rate
        self.timer = self.create_timer(self.control_period, self.control_callback)
        self.time = self.get_clock().now().nanoseconds / 1e9
        self.prev_time = self.time

        self.get_logger().info("LAUV Controller Node Initialized.")

    # --- callbacks ---

    def cmd_odom_callback(self, msg):
        r, p, y = self.euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self.eta_desired[0, 0] = msg.pose.pose.position.x
        self.eta_desired[1, 0] = msg.pose.pose.position.y
        self.eta_desired[2, 0] = msg.pose.pose.position.z
        self.eta_desired[3, 0] = r
        self.eta_desired[4, 0] = p
        self.eta_desired[5, 0] = y

        self.nu_desired[0, 0] = msg.twist.twist.linear.x
        self.nu_desired[1, 0] = msg.twist.twist.linear.y
        self.nu_desired[2, 0] = msg.twist.twist.linear.z
        self.nu_desired[3, 0] = msg.twist.twist.angular.x
        self.nu_desired[4, 0] = msg.twist.twist.angular.y
        self.nu_desired[5, 0] = msg.twist.twist.angular.z
        self.ref_trajectory_received = True

    def odom_callback(self, msg):
        r, p, y = self.euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self.eta_actual[0, 0] = msg.pose.pose.position.x
        self.eta_actual[1, 0] = msg.pose.pose.position.y
        self.eta_actual[2, 0] = msg.pose.pose.position.z
        self.eta_actual[3, 0] = r
        self.eta_actual[4, 0] = p
        self.eta_actual[5, 0] = y

        self.nu_actual[0, 0] = msg.twist.twist.linear.x
        self.nu_actual[1, 0] = msg.twist.twist.linear.y
        self.nu_actual[2, 0] = msg.twist.twist.linear.z
        self.nu_actual[3, 0] = msg.twist.twist.angular.x
        self.nu_actual[4, 0] = msg.twist.twist.angular.y
        self.nu_actual[5, 0] = msg.twist.twist.angular.z
        self.odom_received = True


    def geometric_control(self):
        # calculate position error in ENU frame

        e_z, i_z, d_z = self.pid_z.calculate_error(
            self.eta_desired[2, 0], self.eta_actual[2, 0], self.time
        )

        # compute desired orientation
        pitch_des = self.pid_z.calculate_pid(e_z, i_z, d_z)
        pitch_des = max(-self.desired_pitch_limit, min(pitch_des, self.desired_pitch_limit))

        e_x, i_x, d_x = self.pid_x.calculate_error(self.nu_desired[0, 0], self.nu_actual[0, 0], self.time)
        forward_control = self.pid_x.calculate_pid(e_x, i_x, d_x)

        roll_e_p, roll_e_i, roll_e_d  = self.pid_phi.calculate_error(
            self.eta_desired[3, 0], self.eta_actual[3, 0], self.time
        )
        roll_control = self.pid_phi.calculate_pid(roll_e_p, roll_e_i, roll_e_d)
        
        pitch_e_p, pitch_e_i, pitch_e_d =  self.pid_theta.calculate_error(
            pitch_des, self.eta_actual[4, 0], self.time)
        pitch_control = self.pid_theta.calculate_pid(pitch_e_p, pitch_e_i, pitch_e_d)

        yaw_e_p, yaw_e_i, yaw_e_d = self.pid_psi.calculate_error(self.eta_desired[5, 0], self.eta_actual[5, 0], self.time)
        yaw_control = self.pid_psi.calculate_pid(yaw_e_p, yaw_e_i, yaw_e_d)

        # NED frame
        self.tau.header.stamp = self.get_clock().now().to_msg()
        self.tau.wrench.force.x = forward_control
        self.tau.wrench.force.y = 0.0
        self.tau.wrench.force.z = 0.0
        self.tau.wrench.torque.x = roll_control
        self.tau.wrench.torque.y = -pitch_control
        self.tau.wrench.torque.z = yaw_control
        
        print(-pitch_control)

        print("Error Pitch", pitch_e_p, "Error Z:", e_z, "Desired Z:", self.eta_desired[2, 0], "Current Z:", self.eta_actual[2, 0], "Desired Pitch:", pitch_des, "Current Pitch:", self.eta_actual[4,0])


    def control_callback(self):
        self.time = self.get_clock().now().nanoseconds / 1e9
        if self.time - self.prev_time <= 0:
            self.prev_time = self.time
            return

        # NOTE: add check for ref traject received if needed
        if self.odom_received:
            self.geometric_control()
            self.torque_pub.publish(self.tau)

        self.prev_time = self.time

    # --- helper functions ---
    def _declare_pid_params(self, axis, kp, ki, kd):
        self._declare_and_fill(f"k_p_{axis}", kp, f"KP {axis}")
        self._declare_and_fill(f"k_i_{axis}", ki, f"KI {axis}")
        self._declare_and_fill(f"k_d_{axis}", kd, f"KD {axis}")

    def _declare_and_fill(self, key, default, desc):
        p = self.declare_parameter(key, default, ParameterDescriptor(description=desc))
        self.config[key] = p.value

    def update_control_param(self):
        # Update Outer Loop
        self.pid_x.reconfig_param(
            self.config["k_p_x"], self.config["k_i_x"], self.config["k_d_x"]
        )
        self.pid_phi.reconfig_param(
            self.config["k_p_phi"], self.config["k_i_phi"], self.config["k_d_phi"]
        )
        self.pid_theta.reconfig_param(
            self.config["k_p_theta"], self.config["k_i_theta"], self.config["k_d_theta"]
        )
        self.pid_z.reconfig_param(
            self.config["k_p_z"], self.config["k_i_z"], self.config["k_d_z"]
        )
        self.pid_psi.reconfig_param(
            self.config["k_p_psi"], self.config["k_i_psi"], self.config["k_d_psi"]
        )

    def callback_params(self, data):
        for parameter in data:
            self.config[parameter.name] = parameter.value
        self.update_control_param()
        return SetParametersResult(successful=True)

    def reset_control(self):
        for pid in [
            self.pid_x,
            self.pid_phi,
            self.pid_theta,
            self.pid_z,
            self.pid_psi,
        ]:
            pid.reset_control()

    # --- utils ---
    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z

    def eulerang(self, phi, theta, psi):
        cphi, sphi = math.cos(phi), math.sin(phi)
        cth, sth = math.cos(theta), math.sin(theta)
        cpsi, spsi = math.cos(psi), math.sin(psi)

        if abs(cth) < 0.001:
            return None, None, None

        # Rotation Matrix (Body to World)
        R = np.array(
            [
                [
                    cpsi * cth,
                    -spsi * cphi + cpsi * sth * sphi,
                    spsi * sphi + cpsi * cphi * sth,
                ],
                [
                    spsi * cth,
                    cpsi * cphi + sphi * sth * spsi,
                    -cpsi * sphi + sth * spsi * cphi,
                ],
                [-sth, cth * sphi, cth * cphi],
            ]
        )

        # Angular Transformation (Body rates to Euler rates)
        T = np.array(
            [
                [1, sphi * sth / cth, cphi * sth / cth],
                [0, cphi, -sphi],
                [0, sphi / cth, cphi / cth],
            ]
        )

        J = np.zeros((6, 6))
        J[0:3, 0:3] = R
        J[3:6, 3:6] = T

        return J, R, T


def main(args=None):
    rclpy.init(args=args)
    node = LAUVControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
