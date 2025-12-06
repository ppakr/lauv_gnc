import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    control_node = Node(
        package="lauv_control",
        executable="lauv_control_node",
        name="lauv_controller",
        output="screen",
        parameters=[
            # You can override PID gains here if needed, or load a YAML file
            # {'k_p_z': 2.5},
        ],
    )

    allocator_node = Node(
        package="lauv_control_allocator",
        executable="lauv_control_allocator_node",
        name="lauv_control_allocator",
        output="screen",
    )

    return LaunchDescription([control_node, allocator_node])
