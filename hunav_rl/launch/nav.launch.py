#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get the share directory for your launch package
    package_share = get_package_share_directory("hunav_rl")
    default_params_file = os.path.join(
        package_share, "config", "nav2_params.yaml"
    )

    # Launch configuration variables
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    autostart = LaunchConfiguration("autostart", default="true")
    params_file = LaunchConfiguration(
        "params_file", default=default_params_file
    )

    # Launch the local costmap node (provided by nav2_costmap_2d)
    local_costmap_node = Node(
        package="nav2_costmap_2d",
        executable="nav2_costmap_2d",
        name="costmap",
        namespace="costmap",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
    )

    # Launch the lifecycle manager for the local costmap.
    lifecycle_manager_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_local_costmap",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "autostart": autostart,
                "node_names": ["costmap/costmap"],
            }
        ],
    )

    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation clock if true",
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "autostart",
            default_value="true",
            description="Automatically startup the local costmap",
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="Path to the parameter file for the local costmap",
        )
    )

    # Add nodes to the launch description
    ld.add_action(local_costmap_node)
    ld.add_action(lifecycle_manager_node)

    return ld
