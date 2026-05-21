import os

from os import path
from os import environ
from os import pathsep
from scripts import GazeboRosPaths
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    Shutdown,
    RegisterEventHandler,
    TimerAction,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    PathJoinSubstitution,
    LaunchConfiguration,
    PythonExpression,
    EnvironmentVariable,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import (
    OnExecutionComplete,
    OnProcessExit,
    OnProcessIO,
    OnProcessStart,
    OnShutdown,
)


def generate_launch_description():
    hunav_rl_launch = os.path.join(
        get_package_share_directory("hunav_rl"), "launch"
    )

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    x_pose = LaunchConfiguration("x_pose", default="-1.0")
    y_pose = LaunchConfiguration("y_pose", default="-3.0")

    # log level argument
    declare_log_level = DeclareLaunchArgument(
        "log_level",
        default_value="INFO",
        description="Logging level for all nodes",
    )
    log_level = LaunchConfiguration("log_level")

    # to activate the use of nvidia gpu
    use_nvidia_gpu = [
        "__NV_PRIME_RENDER_OFFLOAD=1 ",
        "__GLX_VENDOR_LIBRARY_NAME=nvidia ",
    ]

    # World generation parameters
    world_file_name = LaunchConfiguration("base_world")
    gz_obs = LaunchConfiguration("use_gazebo_obs")
    rate = LaunchConfiguration("update_rate")
    robot_name = LaunchConfiguration("robot_name")
    global_frame = LaunchConfiguration("global_frame_to_publish")
    use_navgoal = LaunchConfiguration("use_navgoal_to_start")
    insert_collision_cylinder = LaunchConfiguration(
        "insert_collision_cylinder"
    )
    ignore_models = LaunchConfiguration("ignore_models")

    # agent configuration file
    agent_conf_file = PathJoinSubstitution(
        [
            FindPackageShare("hunav_rl"),
            "config",
            LaunchConfiguration("configuration_file"),
        ]
    )

    # Read the yaml file and load the parameters
    hunav_loader_node = Node(
        package="hunav_agent_manager",
        executable="hunav_loader",
        output="screen",
        parameters=[agent_conf_file],
        arguments=["--ros-args", "--log-level", log_level],
    )

    # world base file
    world_file = PathJoinSubstitution(
        [FindPackageShare("hunav_rl"), "worlds", world_file_name]
    )

    hunav_gazebo_worldgen_node = Node(
        package="hunav_gazebo_wrapper",
        executable="hunav_gazebo_world_generator",
        output="screen",
        parameters=[
            {"base_world": world_file},
            {"use_gazebo_obs": gz_obs},
            {"update_rate": rate},
            {"robot_name": robot_name},
            {"global_frame_to_publish": global_frame},
            {"use_navgoal_to_start": use_navgoal},
            {"insert_collision_cylinder": insert_collision_cylinder},
            {"ignore_models": ignore_models},
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )

    ordered_launch_event = RegisterEventHandler(
        OnProcessStart(
            target_action=hunav_loader_node,
            on_start=[
                LogInfo(
                    msg=(
                        "HunNavLoader started, launching "
                        "HuNav_Gazebo_world_generator after 2 seconds..."
                    )
                ),
                TimerAction(
                    period=2.0,
                    actions=[hunav_gazebo_worldgen_node],
                ),
            ],
        )
    )

    # Then, launch the generated world in Gazebo
    my_gazebo_models = PathJoinSubstitution(
        [
            FindPackageShare("hunav_gazebo_wrapper"),
            "models",
        ]
    )

    config_file_name = "params.yaml"
    pkg_dir = get_package_share_directory("hunav_gazebo_wrapper")
    config_file = path.join(pkg_dir, "launch", config_file_name)

    model, plugin, media = GazeboRosPaths.get_paths()
    print("model:", model)
    print("plugin:", plugin)
    print("media:", media)
    print("my_gazebo_models:", my_gazebo_models)

    if "GAZEBO_MODEL_PATH" in environ:
        model += pathsep + environ["GAZEBO_MODEL_PATH"]
    if "GAZEBO_PLUGIN_PATH" in environ:
        plugin += pathsep + environ["GAZEBO_PLUGIN_PATH"]
    if "GAZEBO_RESOURCE_PATH" in environ:
        media += pathsep + environ["GAZEBO_RESOURCE_PATH"]

    env = {
        "GAZEBO_MODEL_PATH": model,
        "GAZEBO_PLUGIN_PATH": plugin,
        "GAZEBO_RESOURCE_PATH": media,
    }
    print("env:", env)

    set_env_gazebo_resource = SetEnvironmentVariable(
        name="GAZEBO_RESOURCE_PATH",
        value=[EnvironmentVariable("GAZEBO_RESOURCE_PATH"), my_gazebo_models],
    )
    set_env_gazebo_plugin = SetEnvironmentVariable(
        name="GAZEBO_PLUGIN_PATH",
        value=[EnvironmentVariable("GAZEBO_PLUGIN_PATH"), plugin],
    )

    world_path = PathJoinSubstitution(
        [
            FindPackageShare("hunav_rl"),
            "worlds",
            "generatedWorld_cafe.world",
        ]
    )

    # Gazebo server
    gzserver_cmd = [
        use_nvidia_gpu,
        "gzserver ",
        world_path,
        _boolean_command("verbose"),
        "",
        "-s ",
        "libgazebo_ros_init.so",
        "-s ",
        "libgazebo_ros_factory.so",
        "-s ",
        "libgazebo_ros_state.so",
        "--ros-args",
        "--params-file",
        config_file,
    ]

    # Gazebo client
    gzclient_cmd = [
        use_nvidia_gpu,
        "gzclient",
        _boolean_command("verbose"),
        " ",
    ]

    gzserver_process = ExecuteProcess(
        cmd=gzserver_cmd,
        output="screen",
        shell=True,
        on_exit=Shutdown(),
    )

    gzclient_process = ExecuteProcess(
        cmd=gzclient_cmd,
        output="screen",
        shell=True,
        on_exit=Shutdown(),
    )

    # Do not launch Gazebo until the world has been generated
    ordered_launch_event2 = RegisterEventHandler(
        OnProcessStart(
            target_action=hunav_gazebo_worldgen_node,
            on_start=[
                LogInfo(
                    msg=("GenerateWorld started, launching Gazebo after 2 "
                        "seconds..."
                    )
                ),
                TimerAction(
                    period=2.0,
                    actions=[
                        gzserver_process,
                    ],
                ),
            ],
        )
    )

    # hunav_manager node
    hunav_manager_node = Node(
        package="hunav_agent_manager",
        executable="hunav_agent_manager",
        name="hunav_agent_manager",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=["--ros-args", "--log-level", log_level],
    )

    metrics_file = PathJoinSubstitution(
        [
            FindPackageShare("hunav_evaluator"),
            "config",
            LaunchConfiguration("metrics_file"),
        ]
    )

    hunav_evaluator_node = Node(
        package="hunav_evaluator",
        executable="hunav_evaluator_node",
        output="screen",
        parameters=[metrics_file],
        arguments=["--ros-args", "--log-level", log_level],
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                hunav_rl_launch, "robot_state_publisher_waffle_rl.launch.py"
            )
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(hunav_rl_launch, "spawn_waffle_rl.launch.py")
        ),
        launch_arguments={"x_pose": x_pose, "y_pose": y_pose}.items(),
    )

    declare_agents_conf_file = DeclareLaunchArgument(
        "configuration_file",
        default_value="agents_cafe.yaml",
        description="Specify configuration file name in the config directory",
    )

    declare_metrics_conf_file = DeclareLaunchArgument(
        "metrics_file",
        default_value="metrics.yaml",
        description=("Specify the name of the metrics configuration file in "
        "the config directory"
        ),
    )

    declare_arg_world = DeclareLaunchArgument(
        "base_world",
        default_value="empty_cafe.world",
        description="Specify world file name",
    )
    declare_gz_obs = DeclareLaunchArgument(
        "use_gazebo_obs",
        default_value="true",
        description=(
            "Whether to fill the agents obstacles with closest "
            "Gazebo obstacle or not"
        ),
    )
    declare_update_rate = DeclareLaunchArgument(
        "update_rate",
        default_value="100.0",
        description="Update rate of the plugin",
    )
    declare_robot_name = DeclareLaunchArgument(
        "robot_name",
        default_value="waffle_rl",
        description="Specify the name of the robot Gazebo model",
    )
    declare_frame_to_publish = DeclareLaunchArgument(
        "global_frame_to_publish",
        default_value="map",
        description=(
            "Name of the global frame in which the position of the "
            "agents are provided"
        ),
    )
    declare_use_navgoal = DeclareLaunchArgument(
        "use_navgoal_to_start",
        default_value="false",
        description=(
            "Whether to start the agents movements when a navigation "
            "goal is received or not"
        ),
    )
    declare_insert_collision_cylinder = DeclareLaunchArgument(
        "insert_collision_cylinder",
        default_value="false",
        description=(
            "Whether to add a collision cylinder for the actors "
            "or not"
        ),
    )
    declare_ignore_models = DeclareLaunchArgument(
        "ignore_models",
        default_value="ground_plane cafe",
        description=(
            "list of Gazebo models that the agents should ignore as obstacles "
            "as the ground_plane. Indicate the models with a blank space "
            "between them"
        ),
    )
    declare_arg_verbose = DeclareLaunchArgument(
        "verbose",
        default_value="false",
        description='Set "true" to increase messages written to terminal.',
    )

    ld = LaunchDescription()
    ld.add_action(declare_log_level)
    ld.add_action(set_env_gazebo_resource)
    ld.add_action(set_env_gazebo_plugin)
    ld.add_action(declare_agents_conf_file)
    ld.add_action(declare_metrics_conf_file)
    ld.add_action(declare_arg_world)
    ld.add_action(declare_gz_obs)
    ld.add_action(declare_update_rate)
    ld.add_action(declare_robot_name)
    ld.add_action(declare_frame_to_publish)
    ld.add_action(declare_use_navgoal)
    ld.add_action(declare_insert_collision_cylinder)
    ld.add_action(declare_ignore_models)
    ld.add_action(declare_arg_verbose)
    ld.add_action(hunav_loader_node)
    ld.add_action(ordered_launch_event)
    ld.add_action(hunav_manager_node)
    ld.add_action(hunav_evaluator_node)
    ld.add_action(ordered_launch_event2)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)

    return ld


# Add boolean commands if true
def _boolean_command(arg):
    cmd = [
        '"--',
        arg,
        '" if "true" == "',
        LaunchConfiguration(arg),
        '" else ""',
    ]
    py_cmd = PythonExpression(cmd)
    return py_cmd
