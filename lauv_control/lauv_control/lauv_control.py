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
        self.get_logger().info("Initializing LAUV Cascade Controller...")

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

        # use only diagonal terms for simplicity
        self.m_rb = np.diag(_m_total)

        # --- PID set up ---
        self.config = {}
        # Position/Orientation Loop Gains
        self._declare_pid_params("x", 1.5, 0.01, 0.1)  # Surge Position
        self._declare_pid_params("z", 2.0, 0.05, 0.1)  # Depth
        self._declare_pid_params("theta", 3.0, 0.1, 0.5)  # Pitch
        self._declare_pid_params("psi", 1.5, 0.0, 0.2)  # Heading

        # Velocity Loop Gains
        self._declare_pid_params("u", 4.0, 1.0, 0.5)  # Surge Velocity
        self._declare_pid_params("q", 5.0, 0.5, 1.0)  # Pitch Rate
        self._declare_pid_params("r", 5.0, 0.5, 1.0)  # Yaw Rate

        # Callback for dynamic reconfigure
        self.add_on_set_parameters_callback(self.callback_params)

        # Outer Loop (Position -> Velocity/Angle)
        self.pid_x = PIDController(type="linear")
        self.pid_z = PIDController(type="linear")  # Output: Desired Pitch (Theta)
        self.pid_theta = PIDController(type="angular")  # Output: Desired Pitch Rate (q)
        self.pid_psi = PIDController(type="angular")  # Output: Desired Yaw Rate (r)

        # Inner Loop (Velocity -> Force/Torque)
        self.pid_u = PIDController(type="linear")  # Output: Force X
        self.pid_q = PIDController(type="linear")  # Output: Torque Y
        self.pid_r = PIDController(type="linear")  # Output: Torque Z

        # Initialize PID values from params
        self.update_control_param()

        # --- ROS 2 interfaces ---
        self.tau = WrenchStamped()
        self.nu_msg = TwistStamped()

        self.odom_sub = self.create_subscription(
            Odometry, "gnc/odom_filtered", self.odom_callback, 10
        )

        self.cmd_pose_sub = self.create_subscription(
            PoseStamped, "gnc/ref_trajectory_filtered", self.cmd_pose_callback, 10
        )

        self.control_switch_service = self.create_service(
            SetBool, "control_switch", self.control_switch_callback
        )

        self.torque_pub = self.create_publisher(
            WrenchStamped, "gnc/cmd_wrench/control", 10
        )

        self.ref_vel_pub = self.create_publisher(TwistStamped, "gnc/cmd_vel", 10)

        # Timing
        self.control_rate = 20.0  # Hz
        self.control_period = 1.0 / self.control_rate
        self.timer = self.create_timer(self.control_period, self.control_callback)
        self.time = self.get_clock().now().nanoseconds / 1e9
        self.prev_time = self.time
        pass

    # --- callbacks ---

    def cmd_pose_callback(self, msg):
        # Convert Quat to Euler
        r, p, y = self.euler_from_quaternion(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )

        # Transform World Position Ref to Body Frame Ref
        # (This calculates the error in the body frame immediately)
        j, j11, j22 = self.eulerang(
            self.eta_actual[3, 0], self.eta_actual[4, 0], self.eta_actual[5, 0]
        )

        if j is None:
            return  # Singularity check

        pos_ref_world = np.array(
            [[msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]]
        ).T

        # We really only care about X (Surge) and Z (Depth) in body frame relative error
        # But commonly, Z is treated in world frame (depth is absolute),
        # while X/Y are relative to the vehicle heading.

        # For simplicity, we store the raw Desired values:
        self.eta_desired[0, 0] = (
            0.0  # Typically X ref is handled by guidance/LOS giving desired velocity,
        )
        # but if doing station keeping:
        # This part depends heavily on if 'pose' is a target point or current setpoint.
        # Let's assume Trajectory Generator gives a 'moving target'

        # CRITICAL LAUV STRATEGY:
        # We don't transform desired Position to Body. We calculate World Error, then transform Error to Body.
        # So here we just store World Desired.
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
        Outer Loop: Calculates desired velocities/rates based on position errors.
        Special Logic for Underactuated LAUV.
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

        # err_body[0] = Surge error (Body X)
        # err_body[1] = Sway error (Body Y) -> IGNORED (Underactuated)
        # err_body[2] = Heave error (Body Z) -> Used for Depth coupling

        # --- Surge Control ---
        e_x, i_x, d_x = self.pid_x.calculate_error(err_body[0, 0], 0.0, self.time)
        self.nu_desired[0, 0] = self.pid_x.calculate_pid(e_x, i_x, d_x)

        # --- Depth Control ---
        e_z, i_z, d_z = self.pid_z.calculate_error(err_body[2, 0], 0.0, self.time)

        # The output of Z-PID is an OFFSET to the desired Pitch
        pitch_offset = self.pid_z.calculate_pid(e_z, i_z, d_z)

        # Total Desired Pitch = Trajectory Pitch + PID Correction

        # NOTE: Inverting sign: If we are too shallow (+Error in NED logic? No, too shallow means Actual < Desired),
        # Desired=10, Actual=0 -> Err=10. We need +Z motion.
        # +Z motion requires Pitch Down (Negative Theta).
        # So we subtract the pitch_offset.
        total_theta_desired = self.eta_desired[4, 0] - pitch_offset

        max_pitch = 45.0 * (np.pi / 180.0)  # limit pitch +/- 45 degrees
        total_theta_desired = np.clip(total_theta_desired, -max_pitch, max_pitch)

        # Pitch Error
        e_theta, i_theta, d_theta = self.pid_theta.calculate_error(
            total_theta_desired, self.eta_actual[4, 0], self.time
        )

        # Output is Desired Pitch Rate (q)
        self.nu_desired[4, 0] = self.pid_theta.calculate_pid(e_theta, i_theta, d_theta)

        # --- Heading Control (Horizontal Plane) ---
        e_psi, i_psi, d_psi = self.pid_psi.calculate_error(
            self.eta_desired[5, 0], self.eta_actual[5, 0], self.time
        )

        self.nu_desired[5, 0] = self.pid_psi.calculate_pid(e_psi, i_psi, d_psi)

        # --- Passive/Uncontrolled DOFs ---
        self.nu_desired[1, 0] = 0.0  # Sway
        self.nu_desired[2, 0] = 0.0  # Heave (Actuated via pitch, so direct ref is 0)
        self.nu_desired[3, 0] = 0.0  # Roll

        # Update message for debugging
        self.nu_msg.header.stamp = self.get_clock().now().to_msg()
        self.nu_msg.twist.linear.x = self.nu_desired[0, 0]
        self.nu_msg.twist.angular.y = self.nu_desired[4, 0]
        self.nu_msg.twist.angular.z = self.nu_desired[5, 0]

    def velocity_control(self):
        """
        Inner Loop: Calculates Forces/Torques based on velocity errors.
        """
        # --- Surge Velocity (u) -> Force X ---
        e_u, i_u, d_u = self.pid_u.calculate_error(
            self.nu_desired[0, 0], self.nu_actual[0, 0], self.time
        )
        acc_u = self.pid_u.calculate_pid(e_u, i_u, d_u)

        # --- Pitch Rate (q) -> Torque Y ---
        e_q, i_q, d_q = self.pid_q.calculate_error(
            self.nu_desired[4, 0], self.nu_actual[4, 0], self.time
        )
        acc_q = self.pid_q.calculate_pid(e_q, i_q, d_q)

        # --- Yaw Rate (r) -> Torque Z ---
        e_r, i_r, d_r = self.pid_r.calculate_error(
            self.nu_desired[5, 0], self.nu_actual[5, 0], self.time
        )
        acc_r = self.pid_r.calculate_pid(e_r, i_r, d_r)

        # Build Acceleration Vector (assuming decoupled for simplicity or simple mass matrix)
        self.acc[0, 0] = acc_u
        self.acc[1, 0] = 0.0
        self.acc[2, 0] = 0.0
        self.acc[3, 0] = 0.0
        self.acc[4, 0] = acc_q
        self.acc[5, 0] = acc_r

        # Calculate Forces/Torques: Tau = M * Acc
        # (Assuming M includes added mass)
        tau_vec = np.matmul(self.m_rb, self.acc)

        self.tau.header.stamp = self.get_clock().now().to_msg()
        self.tau.wrench.force.x = tau_vec[0, 0]
        self.tau.wrench.force.y = 0.0
        self.tau.wrench.force.z = 0.0  # No vertical thruster
        self.tau.wrench.torque.x = 0.0  # No roll actuation (usually)
        self.tau.wrench.torque.y = tau_vec[4, 0]
        self.tau.wrench.torque.z = tau_vec[5, 0]

    def control_callback(self):
        self.time = self.get_clock().now().nanoseconds / 1e9

        # Safety check: if time delta is weird (first loop), skip
        if self.time - self.prev_time <= 0:
            self.prev_time = self.time
            return

        if self.odom_received and self.ref_trajectory_received and self.is_control_on:
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
        self.pid_z.reconfig_param(
            self.config["k_p_z"], self.config["k_i_z"], self.config["k_d_z"]
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

    def control_switch_callback(self, request, response):
        self.is_control_on = request.data
        if not self.is_control_on:
            self.reset_control()
            response.message = "Control Disabled"
        else:
            response.message = "Control Enabled"
        response.success = True
        return response

    def reset_control(self):
        self.pid_x.reset_control()
        self.pid_z.reset_control()
        self.pid_theta.reset_control()
        self.pid_psi.reset_control()
        self.pid_u.reset_control()
        self.pid_q.reset_control()
        self.pid_r.reset_control()

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
            return None, None, None  # Gimbal lock guard

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
