import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Get path to config
    config = os.path.join(
        get_package_share_directory("lauv_control"),
        "config",
        "lauv_control_params.yaml",
    )

    control_node = Node(
        package="lauv_control",
        executable="lauv_control_node",
        name="lauv_controller",
        output="screen",
        parameters=[config],
    )

    allocator_node = Node(
        package="lauv_control_allocator",
        executable="lauv_control_allocator_node",
        name="lauv_control_allocator",
        output="screen",
    )

    return LaunchDescription([control_node, allocator_node])
