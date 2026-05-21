import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    bringup_dir = get_package_share_directory("hunav_rl")
    pkg_share = FindPackageShare("hunav_rl")
    # Nav2 Params
    namespace = ""
    use_sim_time = True
    autostart = "True"
    params_file = (
        get_package_share_directory("hunav_rl") + "/config/nav2_params.yaml"
    )
    remappings = [
        ("/tf", "tf"),
        ("/tf_static", "tf_static"),
        ("scan", "/scan"),
    ]
    use_respawn = False
    qos = LaunchConfiguration("qos")
    env_id_str = LaunchConfiguration("env_id")

    param_substitutions = {"autostart": autostart}

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
        "initial_pose": "1 1 0 0 0 0",
    }

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key=namespace,
            param_rewrites=param_substitutions,
            convert_types=True,
        ),
        allow_substs=True,
    )

    declare_qos_cmd = DeclareLaunchArgument(
        "qos",
        default_value="2",
        description="QoS used for input sensor topics",
    )

    env_id_arg = DeclareLaunchArgument(
        "env_id", default_value="0", description="Id of the environment"
    )

    # Declare launch arguments.
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

    log_level = LaunchConfiguration("log_level")
    num_envs = LaunchConfiguration("num_envs")

    visualisation = Node(
        name="rviz",
        package="rviz2",
        executable="rviz2",
        arguments=[
            "-d",
            os.path.join(
                get_package_share_directory("hunav_rl"),
                "rviz",
                "training.rviz",
            ),
            "--ros-args",
            "--log-level",
            "fatal",
        ],
        output="log",
        parameters=[{"use_sim_time": use_sim_time}],
    )

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
                        "__import__('os').path.expanduser(",
                        "'~/.ros/rtabmap_eng_hall_'+ ",
                        "str(int('",
                        env_id_str,
                        "')) + '.db')",
                    ]
                )
            },
            {"RGBD/PublishOccupancyGrid": True},
            {"use_sim_time": use_sim_time},
        ],
        remappings=remappings,
    )

    nav2_planner = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        respawn=use_respawn,
        respawn_delay=2.0,
        parameters=[configured_params],
        arguments=["--ros-args", "--log-level", "fatal"],
        remappings=remappings,
    )

    nav2_life_cycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        arguments=["--ros-args", "--log-level", "info"],
        parameters=[{"autostart": True}, {"node_names": ["planner_server"]}],
    )

    ld = LaunchDescription()
    ld.add_action(declare_qos_cmd)
    ld.add_action(env_id_arg)
    ld.add_action(log_level_arg)
    ld.add_action(num_envs_arg)
    ld.add_action(visualisation)
    ld.add_action(rtabmap_node)
    ld.add_action(nav2_planner)
    ld.add_action(nav2_life_cycle_manager)
    return ld


if __name__ == "__main__":
    generate_launch_description()
