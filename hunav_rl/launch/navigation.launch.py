
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LoadComposableNodes, SetParameter
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode, ParameterFile
from nav2_common.launch import RewrittenYaml
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch.actions import TimerAction


def generate_launch_description():
    # Get the launch directory
    bringup_dir = get_package_share_directory("hunav_rl")

    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    params_file = LaunchConfiguration("params_file")
    use_composition = LaunchConfiguration("use_composition")
    container_name = LaunchConfiguration("container_name")
    container_name_full = (namespace, "/", container_name)
    use_respawn = LaunchConfiguration("use_respawn")
    log_level = LaunchConfiguration("log_level")
    world_path = LaunchConfiguration("world_path")
    update_rate = LaunchConfiguration("update_rate")
    step_size = LaunchConfiguration("step_size")
    show_viz = LaunchConfiguration("show_viz")
    viz_pub_rate = LaunchConfiguration("viz_pub_rate")

    qos = LaunchConfiguration("qos")
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
        "Mem/IncrementalMemory": "False",
        "Grid/RangeMax": "5.0",
        "Grid/CellSize": "0.1",
        "Grid/3D": "False",
        "DbSqlite3/CacheSize": "2000",
        "Icp/CCSamplingLimit": "50000",
        "Icp/CorrespondenceRatio": "0.2",
        "Icp/DownsamplingStep": "1",
        "Icp/Iterations": "30",
        "Icp/MaxCorrespondenceDistance": "0.1",
        "Icp/PointToPlane": "false",
        "Icp/RangeMax": "0",
        "Icp/Strategy": "0",
        "Mem/InitWMWithAllNodes": "True",
        "Mem/LaserScanDownsampleStepSize": "1",
        "initial_pose": "-1 -3 0 0 0 0",
    }

    lifecycle_nodes = [
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
    ]

    remappings = [
        ("/tf", "tf"),
        ("/tf_static", "tf_static"),
        ("scan", "/scan"),
    ]

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {"autostart": autostart}

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key=namespace,
            param_rewrites=param_substitutions,
            convert_types=True,
        ),
        allow_substs=True,
    )

    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )

    declare_qos_cmd = DeclareLaunchArgument(
        "qos",
        default_value="2",
        description="QoS used for input sensor topics",
    )

    declare_namespace_cmd = DeclareLaunchArgument(
        "namespace", default_value="", description="Top-level namespace"
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(
            bringup_dir,
            "config",
            "nav2_params.yaml",
        ),
        description=(
            "Full path to the ROS2 parameters file to use "
            "for all launched nodes"
        ),
    )

    declare_autostart_cmd = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Automatically startup the nav2 stack",
    )

    declare_use_composition_cmd = DeclareLaunchArgument(
        "use_composition",
        default_value="False",
        description="Use composed bringup if True",
    )

    declare_container_name_cmd = DeclareLaunchArgument(
        "container_name",
        default_value="nav2_container",
        description=(
            "the name of container that nodes will load in "
            "if use composition"
        ),
        )

    declare_use_respawn_cmd = DeclareLaunchArgument(
        "use_respawn",
        default_value="True",
        description=(
            "Whether to respawn if a node crashes. "
            "Applied when composition is disabled."
        ),
    )

    declare_log_level_cmd = DeclareLaunchArgument(
        "log_level", default_value="info", description="log level"
    )

    declare_update_rate_cmd = DeclareLaunchArgument(
        name="update_rate",
        default_value="100.0",
        description="Update rate for the Flatland server",
    )

    declare_step_size_cmd = DeclareLaunchArgument(
        name="step_size",
        default_value="0.01",
        description="Step size for the Flatland server",
    )

    declare_show_viz_cmd = DeclareLaunchArgument(
        name="show_viz",
        default_value="true",
        description="Show visualization for the Flatland server",
    )

    declare_viz_pub_rate_cmd = DeclareLaunchArgument(
        name="viz_pub_rate",
        default_value="30.0",
        description="Visualization publish rate for the Flatland server",
    )

    load_nodes = GroupAction(
        condition=IfCondition(PythonExpression(["not ", use_composition])),
        actions=[
            SetParameter("use_sim_time", use_sim_time),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[
                    configured_params,
                    {
                        "default_nav_to_pose_bt_xml": os.path.join(
                            bringup_dir,
                            "behavior_trees",
                            "navigate_to_pose_w_replanning_and_rl_agent.xml",
                        )
                    },
                ],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_waypoint_follower",
                executable="waypoint_follower",
                name="waypoint_follower",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                arguments=["--ros-args", "--log-level", log_level],
                parameters=[
                    {"autostart": autostart},
                    {"node_names": lifecycle_nodes},
                ],
            ),
        ],
    )
    timer_action = TimerAction(period=2.0, actions=[load_nodes])

    rtabmap_node = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        parameters=[
            parameters,
            {
                "database_path": PythonExpression(
                    [
                        "__import__('os').path.expanduser('~/.ros/rtabmap_hospital.db')"
                    ]
                )
            },
            {"RGBD/PublishOccupancyGrid": True},
            {"use_sim_time": use_sim_time},
        ],
        remappings=remappings,
    )

    rl_follow_path_node = Node(
        package="hunav_rl",
        executable="path_follower",
        name="navigation",
        output="screen",
        respawn=use_respawn,
        respawn_delay=2.0,
    )

    launch_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("nav2_bringup"),
                        "launch",
                        "rviz_launch.py",
                    ]
                )
            ]
        ),
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Set environment variables
    ld.add_action(stdout_linebuf_envvar)

    # Declare the launch options
    ld.add_action(declare_qos_cmd)
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_container_name_cmd)
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(declare_log_level_cmd)
    ld.add_action(declare_update_rate_cmd)
    ld.add_action(declare_step_size_cmd)
    ld.add_action(declare_show_viz_cmd)
    ld.add_action(declare_viz_pub_rate_cmd)
    ld.add_action(rl_follow_path_node)
    ld.add_action(rtabmap_node)
    ld.add_action(timer_action)
    ld.add_action(launch_rviz)

    return ld
