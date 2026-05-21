#!/usr/bin/env python3
"""Entry point script for running robot navigation evaluations in various
environments."""

import argparse
import subprocess
import shutil
import os
import signal
import sys
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from math import sqrt

terminal_processes = []
sim_groups = {}
eval_procs = []
start_ids = []
num_simulations_ = 1
first_domain_id_ = 0
simulation_running_status = None
ws_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",".."))


def open_terminal_with_command(command: str, delay: float, title: str):
    """Open a new gnome-terminal window with specified command and title.

    Args:
        command: Shell command to execute in the terminal.
        delay: Time delay before executing the command.
        title: Terminal window title.

    Returns:
        Subprocess object for the opened terminal.
    """
    # Set terminal title using an ANSI escape code.
    full_command = (
        f'echo -ne "\033]0;{title}\007"; sleep {delay}; {command}; exec bash'
    )
    proc = subprocess.Popen(
        [
            "gnome-terminal",
            "--disable-factory",
            "--",
            "bash",
            "-c",
            full_command,
        ],
        preexec_fn=os.setsid,
    )
    print(f"Opened terminal with command: {full_command} (PID: {proc.pid})")
    return proc


def start_simulation(
    first_domain_id,
    sim_index,
    num_simulations,
    num_evaluations,
    script_dir,
    ros_dir,
    result_file,
    delays,
    world_name,
    use_gzclient,
    num_people,
    planner,
):
    """Start a simulation instance with specified parameters.

    Args:
        first_domain_id: Base ROS domain ID.
        sim_index: Index of this simulation instance.
        num_simulations: Total number of simulation instances.
        num_evaluations: Total number of evaluations to run.
        script_dir: Directory containing scripts.
        ros_dir: ROS workspace directory.
        result_file: Path to result output file.
        delays: Timing delays for simulation startup.
        world_name: Name of the Gazebo world to use.
        use_gzclient: Whether to show Gazebo GUI.
        num_people: Number of people in the simulation.
        planner: Navigation planner to use.
    """
    domain_id = first_domain_id + sim_index
    port = 11345 + sim_index
    start_id = int(sim_index * num_evaluations / num_simulations)

    # Log start simulation
    print(
        f"Sim_index {sim_index} num_simulations_ {num_simulations}; "
        f"100/num_sim {num_evaluations/num_simulations}."
    )

    # Replace RTAB-Map database file for the simulation.
    if world_name == "small_house":
        source = os.path.join(script_dir, "maps", f"rtabmap_{world_name}.db")
        destination = os.path.join(
            ros_dir, f"rtabmap_{world_name}_{sim_index}.db"
        )
        gazebo_launch = (
            f"gazebo_{world_name}.launch.py use_gzclient:={use_gzclient} "
            f"num_people:={num_people} result_file:={result_file}"
        )
        rtabmap_launch = (
            f"rtabmap.launch.py world_name:={world_name} env_id:={sim_index} "
            f"initial_pose:='0 0 0 0 0 0'"
        )
        extra_export = ""
    elif world_name == "small_hospital":
        source = os.path.join(script_dir, "maps", f"rtabmap_{world_name}.db")
        gazebo_launch = (
            f"gazebo_{world_name}.launch.py use_gzclient:={use_gzclient} "
            f"num_people:={num_people} "
            f"result_file:={result_file}"
        )
        destination = os.path.join(
            ros_dir, f"rtabmap_{world_name}_{sim_index}.db"
        )
        rtabmap_launch = (
            f"rtabmap.launch.py world_name:={world_name} "
            f"env_id:={sim_index} "
            "initial_pose:='-1 -3 0 0 0 0'"
        )
        model_path = os.path.join(script_dir, "models", "models_hospital")
        extra_export = (
            f"export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:{model_path};"
        )
    else:
        print(f"ERROR unknown world name {world_name}")
        return
    try:
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o644)
        print(f"Replaced {destination} with {source}")
    except Exception as e:
        print(f"Error replacing {destination} with {source}: {e}")

    group_title = f"Simulation {sim_index}:"

    base_delay = delays[0] * sim_index
    env_cmd = (
        f"export ROS_DOMAIN_ID={domain_id}; "
        f"export GAZEBO_MASTER_URI=http://localhost:{port};"
    )
    cmd_gazebo = (
        f"{env_cmd} {extra_export} ros2 launch hunav_rl {gazebo_launch}"
    )
    cmd_rtabmap = f"{env_cmd} ros2 launch hunav_rl {rtabmap_launch}"
    if (
        planner == "DRL-VO_Reconstruction"
        or planner == "SAC-HuMap-LeNet-Cost2"
        or planner == "SAC-HuMap-LeNet-Cost2_VO"
        or planner == "SAC-HuMap-LeNet-Cost2_SFM"
    ):
        cmd_nav = f"{env_cmd} ros2 launch hunav_rl navigation_rl.launch.py"
    elif planner == "DWA":
        cmd_nav = f"{env_cmd} ros2 launch hunav_rl navigation_dwb.launch.py"
    elif planner == "MPPI":
        cmd_nav = f"{env_cmd} ros2 launch hunav_rl navigation_mppi.launch.py"
    else:
        print(f"ERROR unknown planner {planner}")
        return
    cmd_rviz = f"{env_cmd} ros2 launch nav2_bringup rviz_launch.py"

    if (
        planner == "SAC-HuMap-LeNet-Cost2"
        or planner == "SAC-HuMap-LeNet-Cost2_VO"
        or planner == "SAC-HuMap-LeNet-Cost2_SFM"
    ):
        cmd_path_follower = (
            f"{env_cmd} ros2 run hunav_rl path_follower_drlsf "
            f"--ros-args -p use_sim_time:=true "
            f"-p observation_mode:=humap "
            f"-p planner_model:={planner}"
        )
    elif planner == "DRL-VO_Reconstruction":
        cmd_path_follower = (
            f"{env_cmd} ros2 launch hunav_rl drlvo_retrained.launch.py"
        )

    cmd_eval = (
        f"{env_cmd} ros2 run hunav_rl eval --ros-args "
        f"-p start_id:={start_id} "
        f"-p num_evaluations:={int(num_evaluations/num_simulations)} "
        f"-p world_name:={world_name} "
        f"-p num_people:={num_people} "
        f"-p planner:={planner}"
    )
    proc_gazebo = open_terminal_with_command(
        cmd_gazebo, delay=base_delay, title=group_title + " Gazebo"
    )
    proc_rtabmap = open_terminal_with_command(
        cmd_rtabmap,
        delay=base_delay + delays[1],
        title=group_title + " RTAB-Map",
    )
    proc_nav = open_terminal_with_command(
        cmd_nav,
        delay=base_delay + delays[2],
        title=group_title + " Navigation",
    )
    if (
        planner == "DRL-VO_Reconstruction"
        or planner == "SAC-HuMap-LeNet-Cost2"
        or planner == "SAC-HuMap-LeNet-Cost2_VO"
        or planner == "SAC-HuMap-LeNet-Cost2_SFM"
    ):
        proc_path_follower = open_terminal_with_command(
            cmd_path_follower,
            delay=base_delay + delays[2] + 2,
            title=group_title + " Path Follower",
        )
    proc_eval = open_terminal_with_command(
        cmd_eval,
        delay=base_delay + delays[3],
        title=group_title + " Evaluation",
    )
    proc_rviz = open_terminal_with_command(
        cmd_rviz, delay=base_delay + delays[3], title=group_title + " Rviz"
    )
    eval_procs.append(proc_eval)
    start_ids.append(sim_index * (num_evaluations / num_simulations))
    if (
        planner == "DRL-VO_Reconstruction"
        or planner == "SAC-HuMap-LeNet-Cost2"
        or planner == "SAC-HuMap-LeNet-Cost2_VO"
        or planner == "SAC-HuMap-LeNet-Cost2_SFM"
    ):
        sim_groups[sim_index] = [
            proc_gazebo,
            proc_rtabmap,
            proc_nav,
            proc_path_follower,
            proc_eval,
            proc_rviz,
        ]
    else:
        sim_groups[sim_index] = [
            proc_gazebo,
            proc_rtabmap,
            proc_nav,
            proc_eval,
            proc_rviz,
        ]
    terminal_processes.extend(sim_groups[sim_index])


def check_all_simulation_groups(
    first_domain_id,
    num_simulations,
    num_evaluations,
    args,
    script_dir,
    ros_dir,
    result_file,
    delays,
    world_name,
    use_gzclient,
    num_people,
    planner,
):
    restart_list = []
    all_simulations_running = False

    global simulation_running_status
    if simulation_running_status is None:
        simulation_running_status = [True] * num_simulations

    print("Checking if all simulation groups are running correctly...")

    while not all_simulations_running:
        # Check if all simulation groups are running correctly.
        # Check whether the metrics file already exists - if not all set to False
        runs_per_simulation = int(num_evaluations / num_simulations)
        simulation_running_status = [True] * num_simulations
        evaluation_folder = os.path.join(
            ws_dir,
            "src",
            "drl-sfm",
            "hunav_rl",
            "evaluation",
            f"{world_name}_{num_people}_{planner}",
        )
        for i in range(num_simulations):
            for j in range(runs_per_simulation):
                metrics_steps_file = os.path.join(
                    evaluation_folder,
                    f"metrics_steps_{i*runs_per_simulation+j}.txt",
                )
                print(
                    f"Check if file exists: {metrics_steps_file}"
                )
                print(
                    f"Simulation index: {i},"
                    f" Number of evaluations: {runs_per_simulation}"
                )
                if not os.path.exists(metrics_steps_file):
                    print(
                        f"Metrics file for path {i*runs_per_simulation+j}"
                        f" does not exist."
                    )
                    simulation_running_status[i] = False
                    break
        if all(simulation_running_status):
            print("Metrics files exist.")
            return

        restart_list = []
        for i in range(num_simulations):
            if simulation_running_status[i]:
                continue
            # Set the ROS_DOMAIN_ID for the node.
            os.environ["ROS_DOMAIN_ID"] = str(first_domain_id + i)
            os.environ["GAZEBO_MASTER_URI"] = f"http://localhost:{11345 + i}"
            # Initialize rclpy and create the node.
            rclpy.init(args=sys.argv)
            checker_node = CostmapChecker()
            timeout_sec = 30.0
            start_time = checker_node.get_clock().now().nanoseconds / 1e9
            while (
                checker_node.get_clock().now().nanoseconds / 1e9 - start_time
            ) < timeout_sec:
                rclpy.spin_once(checker_node, timeout_sec=0.5)
                if (
                    checker_node.received_global_costmap
                    and checker_node.received_local_costmap
                    and checker_node.robot_is_moving
                ):
                    break

            if (
                checker_node.received_global_costmap
                and checker_node.received_local_costmap
                and checker_node.robot_is_moving
            ):
                simulation_running_status[i] = True
                print(f"Simulation {i} is running correctly.")
            else:
                restart_list.append(i)
            checker_node.destroy_node()
            rclpy.shutdown()
        if len(restart_list) == 0:
            print("All simulations are running correctly.")
            all_simulations_running = True
            break
        for i in restart_list:
            print(f"Simulation {i} is not running correctly. Terminating...")
            group = sim_groups.get(i, [])
            for proc in group:
                try:
                    if proc.poll() is None:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                except Exception as e:
                    print(f"Error terminating process in simulation {i}: {e}")

        # Restart simulations that are not running correctly.
        for r in range(len(restart_list)):

            start_simulation(
                first_domain_id,
                restart_list[r],
                num_simulations,
                num_evaluations,
                script_dir,
                ros_dir,
                result_file,
                delays,
                world_name,
                use_gzclient,
                num_people,
                planner,
            )

        # Wait until evaluation delay finishes.
        evaluation_delay = (
            delays[0] * (args.num_simulations - 1) + delays[3] + 5
        )
        for remaining in range(evaluation_delay, 0, -1):
            sys.stdout.write("\033]0;Training\007")
            sys.stdout.flush()
            print(f"Ready for evaluation in {remaining} seconds...", end="\r")
            time.sleep(1)


def cleanup_terminals():
    print("Terminating all simulation terminals gracefully...")
    # Kill all terminal processes tracked in the global list.
    for proc in terminal_processes:
        try:
            pgid = os.getpgid(proc.pid)
            print(f"Sending SIGINT to process group with PGID {pgid}")
            os.killpg(pgid, signal.SIGINT)
        except Exception as e:
            print(f"Error sending SIGINT to process group: {e}")
    time.sleep(5)
    for proc in terminal_processes:
        try:
            pgid = os.getpgid(proc.pid)
            print(f"Sending SIGTERM to process group with PGID {pgid}")
            os.killpg(pgid, signal.SIGTERM)
        except Exception as e:
            print(f"Error sending SIGTERM to process group: {e}")
    try:
        subprocess.run(
            ["pkill", "-TERM", "-f", "gnome-terminal --disable-factory"],
            check=True,
        )
        print("Also issued pkill for gnome-terminal as fallback.")
    except Exception as e:
        print(f"Error using pkill: {e}")


def signal_handler(sig, frame):
    print("Ctrl-C pressed, terminating all simulation terminals gracefully...")
    cleanup_terminals()
    sys.exit(0)


class CostmapChecker(Node):
    def __init__(self):
        super().__init__("costmap_checker")
        self.received_global_costmap = False
        self.received_local_costmap = False
        self.robot_is_moving = False
        self.last_position = None
        self.sub_global_costmap = self.create_subscription(
            OccupancyGrid,
            "global_costmap/costmap",
            self.global_costmap_callback,
            10,
        )
        self.sub_local_costmap = self.create_subscription(
            OccupancyGrid,
            "local_costmap/costmap",
            self.local_costmap_callback,
            10,
        )
        self.robot_pose_sub = self.create_subscription(
            Odometry, "odom", self.robot_pose_callback, 10
        )

    def global_costmap_callback(self, msg):
        if not self.received_global_costmap:
            self.received_global_costmap = True

    def local_costmap_callback(self, msg):
        if not self.received_local_costmap:
            self.received_local_costmap = True

    def robot_pose_callback(self, msg):
        if self.last_position is None:
            self.last_position = msg.pose.pose.position
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y

        if self.last_position is not None:
            last_x, last_y = self.last_position.x, self.last_position.y
            distance = sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

            if distance > 0.05:
                self.robot_is_moving = True
            else:
                self.robot_is_moving = False


def main():
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(
        description=(
            "Start simulations in separate terminals with delays and "
            "launch evaluation terminal."
        )
    )
    # Amount of evaluation runs
    parser.add_argument(
        "--num_evaluations",
        type=int,
        default=1,
        help="Number of simulations to launch (default: 100)",
    )
    # Amount of simulations to launch
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=1,
        help="Number of simulations to launch (default: 10)",
    )
    parser.add_argument(
        "--first_ros_domain_id",
        type=int,
        default=60,
        help="Starting ROS_DOMAIN_ID (default: 10)",
    )
    # World names
    # - small_hospital
    # - small_house
    parser.add_argument(
        "--world_name",
        type=str,
        default="small_hospital",
        help="Which world to use for the simulation (default: hospital)",
    )
    # Whether to use gzclient for the simulation
    parser.add_argument(
        "--use_gzclient",
        type=str,
        default="False",
        help="Whether to use gzclient for the simulation (default: True)",
    )
    # Number of people in the simulation
    # - 5
    # - 10
    # - 30
    parser.add_argument(
        "--num_people",
        type=int,
        default=30,
        help="Number of people in the simulation (default: 15)",
    )
    # Planner to use for the simulation
    # - DWA
    # - MPPI
    # - DRL-VO_Reconstruction
    # - SAC-HuMap-LeNet-Cost2
    # - SAC-HuMap-LeNet-Cost2_VO
    # - SAC-HuMap-LeNet-Cost2_SFM
    parser.add_argument(
        "--planner",
        type=str,
        default="SAC-HuMap-LeNet-Cost2_SFM",
        help="Planner to use for the simulation (default: SFM)",
    )
    parser.add_argument(
        "--fast",
        type=bool,
        default=False,
        help="Shorter waiting time between launching simulations",
    )
    args = parser.parse_args()

    # num_simulations_ = args.num_simulations
    first_domain_id = args.first_ros_domain_id
    simulation_running_status = [False] * args.num_simulations

    sys.stdout.write("\033]0;Training\007")
    sys.stdout.flush()

    ros_dir = os.path.join(os.path.expanduser("~"), ".ros")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check whether the evaluation folder already exists and create
    eval_dir = os.path.join(
        ws_dir,
        "src",
        "drl-sfm",
        "hunav_rl",
        "evaluation",
        f"{args.world_name}_{args.num_people}_{args.planner}",
    )
    if not os.path.exists(eval_dir):
        try:
            os.makedirs(eval_dir, exist_ok=True)
            print(f"Created new folder: {eval_dir}")
        except Exception as e:
            print(f"Error creating folder {eval_dir}: {e}")
    else:
        print(f"Evaluation folder already exists: {eval_dir}")
    result_file = os.path.join(eval_dir, "metrics.txt")

    # Copy files from evaluation/eval_path_{world} into the evaluation folder
    source_file = os.path.join(
        ws_dir,
        "src",
        "drl-sfm",
        "hunav_rl",
        "eval_paths",
        f"eval_paths_{args.world_name}.pkl",
    )
    if os.path.exists(source_file):
        destination_file = os.path.join(
            eval_dir, os.path.basename(source_file)
        )
        try:
            shutil.copyfile(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")
        except Exception as e:
            print(f"Error copying {source_file} to {destination_file}: {e}")
    else:
        print(f"Source file {source_file} does not exist.")

    # Copy the agents configuration file into the evaluation folder
    config_file = f"config/agents_{args.world_name}_{args.num_people}.yaml"
    if os.path.exists(config_file):
        destination_config_file = os.path.join(
            eval_dir, os.path.basename(config_file)
        )
        try:
            shutil.copyfile(config_file, destination_config_file)
            print(f"Copied {config_file} to {destination_config_file}")
        except Exception as e:
            print(
                f"Error copying {config_file} to {destination_config_file}: {e}"
            )
    else:
        print(f"Config file {config_file} does not exist.")

    # Replace backup files for simulations before starting terminals.
    files_to_replace = []
    for i in range(args.num_simulations):
        files_to_replace.append(
            (
                f"rtabmap_{args.world_name}.db",
                f"rtabmap_{args.world_name}_{i}.db",
            )
        )
    for backup, original in files_to_replace:
        source = os.path.join(script_dir, "maps", backup)
        destination = os.path.join(ros_dir, original)
        try:
            shutil.copyfile(source, destination)
            os.chmod(destination, 0o644)
            print(f"Replaced {destination} with {source}")
        except Exception as e:
            print(f"Error replacing {destination} with {source}: {e}")

    delays = [5, 20, 40, 60]
    if args.fast == True:
        print("Fast mode is activated")
        delays = [d // 2 for d in delays]
    num_sim = args.num_simulations
    for i in range(num_sim):
        start_simulation(
            first_domain_id,
            i,
            args.num_simulations,
            args.num_evaluations,
            script_dir,
            ros_dir,
            result_file,
            delays,
            args.world_name,
            args.use_gzclient,
            args.num_people,
            args.planner,
        )

    # Wait until evaluation delay finishes.
    evaluation_delay = delays[0] * (args.num_simulations - 1) + delays[3] + 5
    for remaining in range(evaluation_delay, 0, -1):
        sys.stdout.write("\033]0;Training\007")
        sys.stdout.flush()
        print(f"Ready for evaluation in {remaining} seconds...", end="\r")
        time.sleep(1)

    # Check and restart simulation groups
    check_all_simulation_groups(
        first_domain_id,
        args.num_simulations,
        args.num_evaluations,
        args,
        script_dir,
        ros_dir,
        result_file,
        delays,
        args.world_name,
        args.use_gzclient,
        args.num_people,
        args.planner,
    )

    print("Evaluations are running ...")
    # wait for Ctrl-C to terminate all processes
    print("Press Ctrl-C to terminate all processes.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(script_dir, "evaluation")

    try:
        print(f"Number of evaluation processes: {len(eval_procs)}")
        for p in eval_procs:
            print(
                f"Process {p.pid} status: "
                f"{'Running' if p.poll() is None else 'Finished'}"
            )
        # iterate over all simulation groups
        # check if evaluation processes are still running
        while not all(p.poll() is not None for p in eval_procs):
            time.sleep(1)

        print("Evaluation processes finished.")
        cleanup_terminals()
        sys.exit(0)
    except KeyboardInterrupt:
        print(
            "Ctrl-C, terminating all simulation terminals gracefully..."
        )
        cleanup_terminals()
        for proc in terminal_processes:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except Exception as e:
                print(f"Error terminating process: {e}")
        print("All processes terminated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
