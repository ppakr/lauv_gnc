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

        # TODO: parameter setup
        # --- State Variables ---
        self.eta_desired = np.zeros((6, 1))  # [x, y, z, phi, theta, psi]
        self.eta_actual = np.zeros((6, 1))
        self.nu_desired = np.zeros((6, 1))  # [u, v, w, p, q, r]
        self.nu_actual = np.zeros((6, 1))
        self.acc = np.zeros((6, 1))  # Desired accelerations

        self.ref_trajectory_received = False
        self.odom_received = False
        self.is_control_on = True

        # TODO: load mrb
        # --- Mass Matrix ---

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
        self.set_parameters_callback(self.callback_params)

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

    # TODO: utils
