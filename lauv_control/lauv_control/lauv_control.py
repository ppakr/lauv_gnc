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

        # --- Mass Matrix ---
        # (Using the diagonal terms from your previous code)
        mass = 18.0
        Ixx = 2.6244
        Iyy = 3.0618
        Izz = 3.0618

        X_dot_u = 1.0
        Y_dot_v = 16.0
        Z_dot_w = 16.0
        K_dot_p = 0.005
        M_dot_q = 1.3
        N_dot_r = 1.3

        _m_total = [
            mass + X_dot_u,  # Surge (u)
            mass + Y_dot_v,  # Sway  (v)
            mass + Z_dot_w,  # Heave (w)
            Ixx + K_dot_p,  # Roll  (p)
            Iyy + M_dot_q,  # Pitch (q)
            Izz + N_dot_r,  # Yaw   (r)
        ]

        self.m_rb = np.diag(_m_total)

        # --- PID set up ---
        self.config = {}

        # Position/Orientation Loop Gains
        self._declare_pid_params("x", 0.2, 0.0, 0.0)  # Surge Position
        self._declare_pid_params("y", 0.2, 0.0, 0.0)  # Sway Position (New)
        self._declare_pid_params("z", 0.5, 0.01, 0.0)  # Depth Position
        self._declare_pid_params("phi", 1.0, 0.0, 0.0)  # Roll Angle (New)
        self._declare_pid_params("theta", 2.0, 0.0, 0.0)  # Pitch Angle
        self._declare_pid_params("psi", 1.5, 0.0, 0.2)  # Heading Angle

        # Velocity Loop Gains
        self._declare_pid_params("u", 2.0, 0.5, 0.0)  # Surge Velocity
        self._declare_pid_params("v", 2.0, 0.0, 0.0)  # Sway Velocity (New)
        self._declare_pid_params("w", 2.0, 0.0, 0.0)  # Heave Velocity (New)
        self._declare_pid_params("p", 2.0, 0.0, 0.0)  # Roll Rate (New)
        self._declare_pid_params("q", 5.0, 0.5, 1.0)  # Pitch Rate
        self._declare_pid_params("r", 5.0, 0.5, 1.0)  # Yaw Rate

        # Callback for dynamic reconfigure
        self.add_on_set_parameters_callback(self.callback_params)

        # --- PID Instances (Full 6-DOF) ---
        # Outer Loop (Position -> Velocity)
        self.pid_x = PIDController(type="linear")
        self.pid_y = PIDController(type="linear")  # New
        self.pid_z = PIDController(type="linear")
        self.pid_phi = PIDController(type="angular", sat=0.5)  # New
        self.pid_theta = PIDController(type="angular", sat=0.5)
        self.pid_psi = PIDController(type="angular", sat=0.5)

        # Inner Loop (Velocity -> Force/Torque)
        self.pid_u = PIDController(type="linear")
        self.pid_v = PIDController(type="linear")  # New
        self.pid_w = PIDController(type="linear")  # New
        self.pid_p = PIDController(type="linear")  # New
        self.pid_q = PIDController(type="linear")
        self.pid_r = PIDController(type="linear")

        # Initialize PID values from params
        self.update_control_param()

        self.get_logger().info("LAUV Controller Parameters Initialized.")

        # --- ROS 2 interfaces ---
        self.tau = WrenchStamped()
        self.nu_msg = TwistStamped()

        self.odom_sub = self.create_subscription(
            Odometry, "/lauv/odometry", self.odom_callback, 10
        )

        self.cmd_pose_sub = self.create_subscription(
            PoseStamped, "/lauv/ref_trajectory_filtered", self.cmd_pose_callback, 10
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

    def cmd_pose_callback(self, msg):
        r, p, y = self.euler_from_quaternion(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        self.eta_desired[0, 0] = msg.pose.position.x
        self.eta_desired[1, 0] = msg.pose.position.y
        self.eta_desired[2, 0] = msg.pose.position.z
        self.eta_desired[3, 0] = r
        self.eta_desired[4, 0] = p
        self.eta_desired[5, 0] = y
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

    # --- core logic ---
    def position_control(self):
        """
        Outer Loop: Calculates desired velocities based on position errors (6 DOF).
        """
        # Calculate Error in World Frame
        err_world = self.eta_desired[0:3] - self.eta_actual[0:3]

        # Transform Position Error to Body Frame
        _, R_body_to_world, _ = self.eulerang(
            self.eta_actual[3, 0], self.eta_actual[4, 0], self.eta_actual[5, 0]
        )
        if R_body_to_world is None:
            return

        R_world_to_body = np.transpose(R_body_to_world)
        err_body = np.matmul(R_world_to_body, err_world)

        # surge
        e_x, i_x, d_x = self.pid_x.calculate_error(err_body[0, 0], 0.0, self.time)
        self.nu_desired[0, 0] = self.pid_x.calculate_pid(e_x, i_x, d_x)

        # sway
        e_y, i_y, d_y = self.pid_y.calculate_error(err_body[1, 0], 0.0, self.time)
        self.nu_desired[1, 0] = self.pid_y.calculate_pid(e_y, i_y, d_y)

        # heave
        e_z, i_z, d_z = self.pid_z.calculate_error(err_body[2, 0], 0.0, self.time)
        self.nu_desired[2, 0] = self.pid_z.calculate_pid(e_z, i_z, d_z)

        # roll
        e_phi, i_phi, d_phi = self.pid_phi.calculate_error(
            self.eta_desired[3, 0], self.eta_actual[3, 0], self.time
        )
        self.nu_desired[3, 0] = self.pid_phi.calculate_pid(e_phi, i_phi, d_phi)

        # pitch
        e_theta, i_theta, d_theta = self.pid_theta.calculate_error(
            self.eta_desired[4, 0], self.eta_actual[4, 0], self.time
        )
        self.nu_desired[4, 0] = self.pid_theta.calculate_pid(e_theta, i_theta, d_theta)

        # yaw
        e_psi, i_psi, d_psi = self.pid_psi.calculate_error(
            self.eta_desired[5, 0], self.eta_actual[5, 0], self.time
        )
        self.nu_desired[5, 0] = self.pid_psi.calculate_pid(e_psi, i_psi, d_psi)

        # Debug Pub
        self.nu_msg.header.stamp = self.get_clock().now().to_msg()
        self.nu_msg.twist.linear.x = self.nu_desired[0, 0]
        self.nu_msg.twist.linear.y = self.nu_desired[1, 0]
        self.nu_msg.twist.linear.z = self.nu_desired[2, 0]
        self.nu_msg.twist.angular.x = self.nu_desired[3, 0]
        self.nu_msg.twist.angular.y = self.nu_desired[4, 0]
        self.nu_msg.twist.angular.z = self.nu_desired[5, 0]

    def velocity_control(self):
        """
        Inner Loop: Calculates Forces/Torques based on velocity errors (6 DOF).
        """
        # surge
        e_u, i_u, d_u = self.pid_u.calculate_error(
            self.nu_desired[0, 0], self.nu_actual[0, 0], self.time
        )
        self.acc[0, 0] = self.pid_u.calculate_pid(e_u, i_u, d_u)

        # sway
        e_v, i_v, d_v = self.pid_v.calculate_error(
            self.nu_desired[1, 0], self.nu_actual[1, 0], self.time
        )
        self.acc[1, 0] = self.pid_v.calculate_pid(e_v, i_v, d_v)

        # heave
        e_w, i_w, d_w = self.pid_w.calculate_error(
            self.nu_desired[2, 0], self.nu_actual[2, 0], self.time
        )
        self.acc[2, 0] = self.pid_w.calculate_pid(e_w, i_w, d_w)

        # roll
        e_p, i_p, d_p = self.pid_p.calculate_error(
            self.nu_desired[3, 0], self.nu_actual[3, 0], self.time
        )
        self.acc[3, 0] = self.pid_p.calculate_pid(e_p, i_p, d_p)

        # pitch
        e_q, i_q, d_q = self.pid_q.calculate_error(
            self.nu_desired[4, 0], self.nu_actual[4, 0], self.time
        )
        self.acc[4, 0] = self.pid_q.calculate_pid(e_q, i_q, d_q)

        # yaw
        e_r, i_r, d_r = self.pid_r.calculate_error(
            self.nu_desired[5, 0], self.nu_actual[5, 0], self.time
        )
        self.acc[5, 0] = self.pid_r.calculate_pid(e_r, i_r, d_r)

        # Calculate Forces/Torques: Tau = M * Acc
        tau_vec = np.matmul(self.m_rb, self.acc)

        self.tau.header.stamp = self.get_clock().now().to_msg()
        self.tau.wrench.force.x = tau_vec[0, 0]
        self.tau.wrench.force.y = tau_vec[1, 0]
        self.tau.wrench.force.z = tau_vec[2, 0]
        self.tau.wrench.torque.x = tau_vec[3, 0]
        self.tau.wrench.torque.y = tau_vec[4, 0]
        self.tau.wrench.torque.z = tau_vec[5, 0]

    def control_callback(self):
        self.time = self.get_clock().now().nanoseconds / 1e9
        if self.time - self.prev_time <= 0:
            self.prev_time = self.time
            return

        # NOTE: add check for ref traject received if needed
        if self.odom_received:
            self.position_control()
            self.velocity_control()
            self.ref_vel_pub.publish(self.nu_msg)
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
        self.pid_y.reconfig_param(
            self.config["k_p_y"], self.config["k_i_y"], self.config["k_d_y"]
        )
        self.pid_z.reconfig_param(
            self.config["k_p_z"], self.config["k_i_z"], self.config["k_d_z"]
        )
        self.pid_phi.reconfig_param(
            self.config["k_p_phi"], self.config["k_i_phi"], self.config["k_d_phi"]
        )
        self.pid_theta.reconfig_param(
            self.config["k_p_theta"], self.config["k_i_theta"], self.config["k_d_theta"]
        )
        self.pid_psi.reconfig_param(
            self.config["k_p_psi"], self.config["k_i_psi"], self.config["k_d_psi"]
        )

        # Update Inner Loop
        self.pid_u.reconfig_param(
            self.config["k_p_u"], self.config["k_i_u"], self.config["k_d_u"]
        )
        self.pid_v.reconfig_param(
            self.config["k_p_v"], self.config["k_i_v"], self.config["k_d_v"]
        )
        self.pid_w.reconfig_param(
            self.config["k_p_w"], self.config["k_i_w"], self.config["k_d_w"]
        )
        self.pid_p.reconfig_param(
            self.config["k_p_p"], self.config["k_i_p"], self.config["k_d_p"]
        )
        self.pid_q.reconfig_param(
            self.config["k_p_q"], self.config["k_i_q"], self.config["k_d_q"]
        )
        self.pid_r.reconfig_param(
            self.config["k_p_r"], self.config["k_i_r"], self.config["k_d_r"]
        )

    def callback_params(self, data):
        for parameter in data:
            self.config[parameter.name] = parameter.value
        self.update_control_param()
        return SetParametersResult(successful=True)

    def reset_control(self):
        for pid in [
            self.pid_x,
            self.pid_y,
            self.pid_z,
            self.pid_phi,
            self.pid_theta,
            self.pid_psi,
            self.pid_u,
            self.pid_v,
            self.pid_w,
            self.pid_p,
            self.pid_q,
            self.pid_r,
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
