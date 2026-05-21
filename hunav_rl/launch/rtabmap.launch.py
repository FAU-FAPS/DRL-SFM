import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    RegisterEventHandler,
    TimerAction,
    LogInfo,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import (
    OnExecutionComplete,
    OnProcessExit,
    OnProcessIO,
    OnProcessStart,
    OnShutdown,
)
from launch.substitutions import PythonExpression


def generate_launch_description():

    hunav_rl_dir = get_package_share_directory("hunav_rl")

    use_sim_time = LaunchConfiguration("use_sim_time")
    qos = LaunchConfiguration("qos")
    localization = LaunchConfiguration("localization")
    navigation = LaunchConfiguration("navigation")
    rviz = LaunchConfiguration("rviz")

    x_goal_pose = LaunchConfiguration("x_pose", default="5.0")
    y_goal_pose = LaunchConfiguration("y_pose", default="1.5")
    env_id = LaunchConfiguration("env_id", default="0")
    world_name = LaunchConfiguration("world_name", default="0")
    parameters = {
        "frame_id": "base_footprint",
        "use_sim_time": use_sim_time,
        "subscribe_depth": False,
        "subscribe_rgb": False,
        "subscribe_scan": True,
        "approx_sync": True,
        "use_action_for_goal": True,
        "qos_scan": qos,
        "qos_imu": qos,
        "Reg/Strategy": "1",
        "Reg/Force3DoF": "true",
        "RGBD/NeighborLinkRefining": "True",
        "Grid/RangeMin": "0.2",
        "Optimizer/GravitySigma": "0",
        "Mem/IncrementalMemory": "True",
        "Grid/RangeMax": "5.0",
        "Grid/CellSize": "0.05",
        "Grid/3D": "False",
        "Icp/CCSamplingLimit": "50000",
        "Icp/CorrespondenceRatio": "0.2",
        "Icp/DownsamplingStep": "1",
        "Icp/Iterations": "30",
        "Icp/MaxCorrespondenceDistance": "0.1",
        "Icp/PointToPlane": "false",
        "Icp/RangeMax": "0",
        "Icp/Strategy": "0",
        "Mem/InitWMWithAllNodes": "false",
        "Mem/LaserScanDownsampleStepSize": "1",
        "initial_pose": LaunchConfiguration("initial_pose"),
        "database_path": LaunchConfiguration("database_path"),
    }

    remappings = [("scan", "/scan")]

    # Launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    declare_qos_cmd = DeclareLaunchArgument(
        "qos",
        default_value="2",
        description="QoS used for input sensor topics",
    )

    env_id_arg = DeclareLaunchArgument(
        "env_id", default_value="0", description="Id of the environment"
    )

    world_name_arg = DeclareLaunchArgument(
        "world_name",
        default_value="hospital",
        description="Name of world to load, e.g., hospital, cafe, lobby, etc.",
    )

    declare_localization_cmd = DeclareLaunchArgument(
        "localization",
        default_value="true",
        description="Launch in localization mode.",
    )

    declare_navigation_cmd = DeclareLaunchArgument(
        "navigation",
        default_value="false",
        description="Launch nav2 navigation stack.",
    )

    declare_rviz_cmd = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="Launch rviz for navigating.",
    )

    declare_database_path = DeclareLaunchArgument(
        "database_path",
        default_value=PythonExpression(
            [
                "'~/.ros/rtabmap_",
                world_name,
                "_'",
                " + ",
                "str(",
                env_id,
                ")",
                " + ",
                "'.db'",
            ]
        ),
        description="Where is the map saved/loaded.",
    )

    declare_initial_pose = DeclareLaunchArgument(
        "initial_pose",
        default_value="1 1 0 0 0 0",
        description="Initial pose for rtabmap",
    )

    # Load rtab-map
    launch_rtab_map = GroupAction(
        actions=[
            Node(
                condition=UnlessCondition(localization),
                package="rtabmap_slam",
                executable="rtabmap",
                output="log",
                parameters=[parameters],
                remappings=remappings,
                arguments=["-d", "--ros-args", "--log-level", "info"],
            ),  # This will delete the previous database (~/.ros/rtabmap.db)
            Node(
                condition=IfCondition(localization),
                package="rtabmap_slam",
                executable="rtabmap",
                output="log",
                parameters=[
                    parameters,
                    {
                        "Mem/IncrementalMemory": "False",
                        "Mem/InitWMWithAllNodes": "True",
                    },
                ],
                remappings=remappings,
                arguments=["--ros-args", "--log-level", "info"],
            ),
        ]
    )

    ld = LaunchDescription()

    # # Declare the launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_qos_cmd)
    ld.add_action(env_id_arg)
    ld.add_action(world_name_arg)
    ld.add_action(declare_localization_cmd)
    ld.add_action(declare_navigation_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_database_path)
    ld.add_action(declare_initial_pose)
    ld.add_action(launch_rtab_map)

    return ld
