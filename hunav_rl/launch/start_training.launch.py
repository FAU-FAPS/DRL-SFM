import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    Shutdown,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_unpause_nodes(context, *args, **kwargs):
    num_envs_str = LaunchConfiguration("num_envs").perform(context)
    num_envs = int(num_envs_str)
    nodes = []
    for i in range(num_envs):
        nodes.append(
            Node(
                package="hunav_rl",
                executable="unpause",
                name=(
                    f"clock_monitor_{i + int(LaunchConfiguration('first_ros_domain_id').perform(context))}"
                ),
                output="screen",
                arguments=[
                    "--domain",
                    str(
                        i
                        + int(
                            LaunchConfiguration("first_ros_domain_id").perform(
                                context
                            )
                        )
                    ),
                ],
            )
        )
    return nodes


def generate_launch_description():
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="INFO",
        description="Logging level for nodes",
    )
    num_envs_arg = DeclareLaunchArgument(
        "num_envs",
        default_value="4",
        description="Number of environments to launch",
    )
    continue_training_arg = DeclareLaunchArgument(
        "continue_training",
        default_value="false",
        description="Whether to continue training from a previous checkpoint",
    )
    first_ros_domain_id_arg = DeclareLaunchArgument(
        "first_ros_domain_id",
        default_value="10",
        description="First ROS domain id to use",
    )

    log_level = LaunchConfiguration("log_level")
    num_envs = LaunchConfiguration("num_envs")
    continue_training = LaunchConfiguration("continue_training")
    first_ros_domain_id = LaunchConfiguration("first_ros_domain_id")

    start_training = Node(
        package="hunav_rl",
        executable="start_training",
        parameters=[
            {
                "use_sim_time": True,
                "num_envs": num_envs,
                "continue_training": continue_training,
                "first_ros_domain_id": first_ros_domain_id,
            }
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )

    ld = LaunchDescription()
    ld.add_action(log_level_arg)
    ld.add_action(num_envs_arg)
    ld.add_action(first_ros_domain_id_arg)
    ld.add_action(continue_training_arg)
    ld.add_action(start_training)
    ld.add_action(OpaqueFunction(function=generate_unpause_nodes))
    ld.add_action(
        RegisterEventHandler(
            OnProcessExit(target_action=start_training, on_exit=[Shutdown()])
        )
    )
    return ld


if __name__ == "__main__":
    generate_launch_description()
