#!/usr/bin/env python3
"""Entry point script for starting reinforcement learning training sessions."""

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
import json
from ament_index_python.packages import get_package_share_directory

terminal_processes = []
sim_groups = {}
use_nextcloud = False
pkg_share_dir = get_package_share_directory("hunav_rl")
ws_dir = os.path.abspath(os.path.join(pkg_share_dir, "..", "..", "..", ".."))


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
    full_command = f'echo -ne "\033]0;{title}\007"; sleep {delay}; {command}'
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
    time.sleep(3)
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


def check_and_restart_simulation(
    sim_index,
    domain_id,
    port,
    extra_export,
    script_dir,
    delays,
    drlvo_observation,
):
    # Determine simulation type and delays for group sim_index.
    reset = ""
    if sim_index % 4 == 0:
        # Cafe simulation
        gazebo_launch = "tb3_cafe_light.launch.py"
        nav2_launch = f"nav_2_cafe.launch.py env_id:={sim_index}"
        group_title = f"Simulation {sim_index}: Cafe"
    elif sim_index % 4 == 1:
        # Hall simulation
        gazebo_launch = "eng_hall.launch.py"
        nav2_launch = f"nav_2_eng_hall.launch.py env_id:={sim_index}"
        reset = "ros2 service call /reset_simulation std_srvs/srv/Empty '{}';"
        group_title = f"Simulation {sim_index}: Hall"
    elif sim_index % 4 == 2:
        # Random world simulation
        gazebo_launch = "random_light.launch.py"
        nav2_launch = f"nav_2_random.launch.py env_id:={sim_index}"
        group_title = f"Simulation {sim_index}: Random World"
    else:
        # Hospital simulation
        gazebo_launch = "hospital_light.launch.py"
        nav2_launch = f"nav_2_hospital.launch.py env_id:={sim_index}"
        model_path = os.path.join(script_dir, "models", "models_hospital")
        extra_export = (
            f"export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:{model_path};"
        )
        group_title = f"Simulation {sim_index}: Hospital"

    env_cmd = f"export ROS_DOMAIN_ID={domain_id}; "
    env_cmd += f"export GAZEBO_MASTER_URI=http://localhost:{port};"
    cmd_gazebo = (
        f"{env_cmd} {extra_export} ros2 launch hunav_rl {gazebo_launch}"
    )
    cmd_nav2 = f"{env_cmd} ros2 launch hunav_rl {nav2_launch}"
    if drlvo_observation:
        cmd_nav = (
            f"{env_cmd} ros2 launch  drl_vo nav_cnn_data.launch.py; {reset}"
        )
    else:
        cmd_nav = f"{env_cmd} ros2 launch hunav_rl nav.launch.py; {reset}"

    # Terminate any dead processes in the group before restarting.
    group = sim_groups.get(sim_index, [])
    for proc in group:
        try:
            if proc.poll() is None:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
        except Exception as e:
            print(f"Error terminating process in simulation {sim_index}: {e}")
    # Re-launch simulation terminals with zero delay.
    new_group = []
    new_proc = open_terminal_with_command(
        cmd_gazebo, delay=0, title=group_title
    )
    new_group.append(new_proc)
    terminal_processes.append(new_proc)
    new_proc = open_terminal_with_command(
        cmd_nav2, delay=delays[1], title=group_title
    )
    new_group.append(new_proc)
    terminal_processes.append(new_proc)
    new_proc = open_terminal_with_command(
        cmd_nav, delay=delays[2], title=group_title
    )
    new_group.append(new_proc)
    terminal_processes.append(new_proc)
    sim_groups[sim_index] = new_group


def check_all_simulation_groups(
    num_simulations, args, script_dir, delays, drlvo_observation
):
    restarted_simulations = 0
    for i in range(num_simulations):
        group = sim_groups.get(i, [])
        group_alive = (
            all(proc.poll() is None for proc in group) if group else False
        )
        if not group_alive:
            domain_id = args.first_ros_domain_id + i
            port = 11345 + i
            print(
                f"Simulation group {i} is not fully running."
            )
            check_and_restart_simulation(
                i, domain_id, port, "", script_dir, delays, drlvo_observation
            )
            restarted_simulations += 1
        else:
            print(f"Simulation group {i} is running.")
    if restarted_simulations > 0:
        return (restarted_simulations - 1) * delays[0] + delays[3]
    else:
        return 0


class CostmapChecker(Node):
    def __init__(self, drlvo_observation):
        super().__init__("costmap_checker")
        self.received_global_costmap = False
        if drlvo_observation:
            self.received_local_costmap = True
        else:
            self.received_local_costmap = False
        self.sub_global_costmap = self.create_subscription(
            OccupancyGrid,
            "global_costmap/costmap",
            self.global_costmap_callback,
            10,
        )
        if drlvo_observation == False:
            self.sub_local_costmap = self.create_subscription(
                OccupancyGrid,
                "costmap/costmap",
                self.local_costmap_callback,
                10,
            )

    def global_costmap_callback(self, msg):
        if not self.received_global_costmap:
            self.received_global_costmap = True

    def local_costmap_callback(self, msg):
        if not self.received_local_costmap:
            self.received_local_costmap = True


def train(main_args, drlvo_observation):
    signal.signal(signal.SIGINT, signal_handler)

    args = main_args

    sys.stdout.write("\033]0;Training\007")
    sys.stdout.flush()

    ros_dir = os.path.join(os.path.expanduser("~"), ".ros")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Replace backup files for simulations before starting terminals.
    files_to_replace = []
    for i in range(args.num_simulations):
        if i % 4 == 0:
            files_to_replace.append(
                ("rtabmap_cafe_backup.db", f"rtabmap_cafe_{i}.db")
            )
        elif i % 4 == 1:
            files_to_replace.append(
                ("rtabmap_eng_hall.db", f"rtabmap_eng_hall_{i}.db")
            )
        elif i % 4 == 2:
            files_to_replace.append(
                ("rtabmap_random.db", f"rtabmap_random_{i}.db")
            )
        elif i % 4 == 3:
            files_to_replace.append(
                ("rtabmap_hospital_backup.db", f"rtabmap_hospital_{i}.db")
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

    delays = [5, 30, 80, 120]
    if args.fast == True:
        delays = [d // 2 for d in delays]

    num_sim = args.num_simulations
    for i in range(num_sim):
        domain_id = args.first_ros_domain_id + i
        port = 11345 + i

        if i % 4 == 0:
            gazebo_launch = "tb3_cafe_light.launch.py"
            nav2_launch = f"nav_2_cafe.launch.py env_id:={i}"
            extra_export = ""
            reset_gazebo = ""
            group_title = f"Simulation {i}: Cafe"
        elif i % 4 == 1:
            gazebo_launch = "eng_hall.launch.py"
            nav2_launch = f"nav_2_eng_hall.launch.py env_id:={i}"
            reset_gazebo = (
                "ros2 service call /reset_simulation std_srvs/srv/Empty '{}';"
            )
            group_title = f"Simulation {i}: Hall"
        elif i % 4 == 2:
            gazebo_launch = "random_light.launch.py"
            nav2_launch = f"nav_2_random.launch.py env_id:={i}"
            extra_export = ""
            reset_gazebo = ""
            group_title = f"Simulation {i}: Random World"
        else:
            gazebo_launch = "hospital_light.launch.py"
            nav2_launch = f"nav_2_hospital.launch.py env_id:={i}"
            model_path = os.path.join(script_dir, "models", "models_hospital")
            extra_export = (
                f"export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:{model_path};"
            )
            reset_gazebo = ""
            group_title = f"Simulation {i}: Hospital"

        base_delay = delays[0] * i
        env_cmd = f"export ROS_DOMAIN_ID={domain_id}; "
        env_cmd += f"export GAZEBO_MASTER_URI=http://localhost:{port};"
        cmd_gazebo = (
            f"{env_cmd} {extra_export} ros2 launch hunav_rl {gazebo_launch}"
        )
        cmd_nav2 = f"{env_cmd} ros2 launch hunav_rl {nav2_launch}"
        if drlvo_observation:
            cmd_nav = f"{env_cmd} ros2 launch  drl_vo nav_cnn_data.launch.py; "
            cmd_nav += f"{reset_gazebo}"
        else:
            cmd_nav = (
                f"{env_cmd} ros2 launch hunav_rl nav.launch.py; {reset_gazebo}"
            )
        proc_gazebo = open_terminal_with_command(
            cmd_gazebo, delay=base_delay, title=group_title
        )
        proc_nav2 = open_terminal_with_command(
            cmd_nav2, delay=base_delay + delays[1], title=group_title
        )
        proc_nav = open_terminal_with_command(
            cmd_nav, delay=base_delay + delays[2], title=group_title
        )
        sim_groups[i] = [proc_gazebo, proc_nav2, proc_nav]
        terminal_processes.extend(sim_groups[i])

    # Wait until training delay finishes.
    training_delay = delays[0] * (args.num_simulations - 1) + delays[3]
    for remaining in range(training_delay, 0, -1):
        sys.stdout.write("\033]0;Training\007")
        sys.stdout.flush()
        print(f"Ready for training in {remaining} seconds...", end="\r")
        time.sleep(1)

    while True:
        restart_list = []
        for i in range(num_sim):
            os.environ["ROS_DOMAIN_ID"] = str(args.first_ros_domain_id + i)
            os.environ["GAZEBO_MASTER_URI"] = f"http://localhost:{11345 + i}"
            rclpy.init(args=sys.argv)
            checker_node = CostmapChecker(drlvo_observation)
            timeout_sec = 40.0
            start_time = checker_node.get_clock().now().nanoseconds / 1e9
            while (
                checker_node.get_clock().now().nanoseconds / 1e9 - start_time
            ) < timeout_sec:
                rclpy.spin_once(checker_node, timeout_sec=0.5)
                if (
                    checker_node.received_global_costmap
                    and checker_node.received_local_costmap
                ):
                    break

            if (
                checker_node.received_global_costmap
                and checker_node.received_local_costmap
            ):
                print(f"Simulation {i} is running correctly.")
            else:
                restart_list.append(i)
            checker_node.destroy_node()
            rclpy.shutdown()
        if len(restart_list) == 0:
            print("All simulations are running correctly.")
            break
        for i in restart_list:
            print(f"Simulation {i} is not running correctly. Restarting...")
            group = sim_groups.get(i, [])
            for proc in group:
                try:
                    if proc.poll() is None:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                except Exception as e:
                    print(f"Error terminating process in simulation {i}: {e}")
        training_delay = check_all_simulation_groups(
            num_sim, args, script_dir, delays, drlvo_observation
        )
        for remaining in range(training_delay, 0, -1):
            sys.stdout.write("\033]0;Training\007")
            sys.stdout.flush()
            print(f"Ready for training in {remaining} seconds...", end="\r")
            time.sleep(1)

    # Launch training.
    training_cmd = (
        f"ros2 launch hunav_rl start_training.launch.py "
        f"num_envs:={args.num_simulations} "
        f"first_ros_domain_id:={args.first_ros_domain_id}"
    )
    if args.continue_training:
        training_cmd += " continue_training:=true"
    print("Launching training now...")
    result = subprocess.run(training_cmd, shell=True)

    print("Training command exited. Cleaning up simulation terminals...")
    cleanup_terminals()

def main():
    parser = argparse.ArgumentParser(
        description="Start simulations and training with scenario management."
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        default=False,
        help="Continue training from a previous checkpoint",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=8,
        help="Number of simulations to launch (default: 8)",
    )
    parser.add_argument(
        "--first_ros_domain_id",
        type=int,
        default=10,
        help="Starting ROS_DOMAIN_ID (default: 10)",
    )
    parser.add_argument(
        "--fast",
        type=bool,
        default=True,
        help="Shorter waiting time between launching simulations",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(script_dir, "evaluation", "eval")
    scenario_file = os.path.join(script_dir, "config", "training_queue.txt")
    scenario_json_file = os.path.join(
        script_dir, "config", "training_configurations.json"
    )
    scenario_config_file = os.path.join(
        script_dir, "config", "training_scenario.json"
    )
    models_dir = os.path.join(script_dir, "hunav_rl", "rl_models")
    logs_dir = os.path.join(script_dir, "hunav_rl", "logs")

    while True:
        # Clean old logs and models only if not continuing training
        if not args.continue_training:
            if os.path.exists(models_dir):
                shutil.rmtree(models_dir)
            if os.path.exists(logs_dir):
                shutil.rmtree(logs_dir)
        else:
            print(
                "Skipping cleanup of logs and models (continue_training=True)"
            )

        # kill old gzserver processes
        subprocess.run(["pkill", "-f", "gzserver"], check=False)
        subprocess.run(["colcon", "build"], cwd=ws_dir, check=True)

        # Get scenario: from existing config file if continuing training, 
        # otherwise from scenarios file
        if args.continue_training:
            # Read scenario from existing scenario config file
            if os.path.exists(scenario_config_file):
                with open(scenario_config_file, "r") as file:
                    scenario_config = json.load(file)
                scenario = scenario_config.get("name", "")
                if scenario:
                    print(f"Continuing training for scenario: {scenario}")
                else:
                    print("No scenario name found in config file. Exit")
                    sys.exit(1)
            else:
                print(
                    "Scenario config file not found for continue training. Exit"
                )
                sys.exit(1)
        else:
            # Read the first line from the scenarios file to determine
            # which training scenario to use and delete this line
            if os.path.exists(scenario_file):
                with open(scenario_file, "r") as file:
                    scenario = file.readline().strip()
                if scenario:
                    print(f"Starting training for scenario: {scenario}")
                    with open(scenario_file, "r+") as file:
                        lines = file.readlines()
                        if len(lines) > 1:
                            file.seek(0)
                            file.writelines(lines[1:])
                            file.truncate()
                        else:
                            os.remove(scenario_file)
                else:
                    print("Scenario file is empty. Exit")
                    sys.exit(1)
            else:
                print("Scenario file not found. Exit")
                sys.exit(1)
        # Read the scenario description from the JSON file
        if os.path.exists(scenario_json_file):
            with open(scenario_json_file, "r") as file:
                scenarios = json.load(file)
            if scenario in scenarios:
                scenario_data = scenarios[scenario]
                print("Scenario data:" + str(scenario_data))
            else:
                print(
                    f"Scenario {scenario} not found in JSON file. "
                    f"Continuing to next scenario."
                )
                continue
        else:
            print("Scenario JSON file not found. Exit")
            sys.exit(1)
        # Create the scenario config file for the current training
        scenario_config = {"name": scenario, "scenario": scenario_data}
        with open(scenario_config_file, "w") as file:
            json.dump(scenario_config, file, indent=4)
        if "drlvo_observation" in scenario_data:
            drlvo_observation = True
        else:
            drlvo_observation = False
        train(args, drlvo_observation)
        print("Training completed. Starting evaluation...")
        if "evaluate" in scenario_data:
            evaluate = True
            print("Evaluation is enabled. Starting evaluation...")
            subprocess.run(["pkill", "-f", "gzserver"], check=False)
            os.makedirs(eval_dir, exist_ok=True)
            for item in os.listdir(eval_dir):
                item_path = os.path.join(eval_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            subprocess.run(
                ["python3 start_evaluation.py"],
                cwd=script_dir,
                shell=True,
                check=True,
            )
            print("Evaluation completed. Moving files...")
            # Rename the evaluation folder to the scenario name
            new_eval_dir = os.path.join(os.path.dirname(eval_dir), scenario)
            if os.path.exists(new_eval_dir):
                new_eval_dir += "_new"
            os.rename(eval_dir, new_eval_dir)
            print(f"Renamed evaluation folder to {new_eval_dir}")
        else:
            evaluate = False
        scenario_dir = os.path.join(script_dir, "scenarios", scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        if use_nextcloud:
            nextcloud_dir = os.path.join(
                os.path.expanduser("~"), "Nextcloud", "scenarios", scenario
            )
            os.makedirs(nextcloud_dir, exist_ok=True)

        # Move models_dir and logs_dir folders to the scenario folder
        for item in os.listdir(models_dir):
            src = os.path.join(models_dir, item)
            dst = os.path.join(scenario_dir, "rl_models", item)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            if use_nextcloud:
                dst_nextcloud = os.path.join(nextcloud_dir, "rl_models", item)
                os.makedirs(os.path.dirname(dst_nextcloud), exist_ok=True)
                shutil.move(src, dst_nextcloud)
        for item in os.listdir(logs_dir):
            src = os.path.join(logs_dir, item)
            dst = os.path.join(scenario_dir, "logs", item)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            if use_nextcloud:
                dst_nextcloud = os.path.join(nextcloud_dir, "logs", item)
                os.makedirs(os.path.dirname(dst_nextcloud), exist_ok=True)
                shutil.move(src, dst_nextcloud)

        if evaluate and use_nextcloud:
            # copy new_eval_dir to Nextcloud
            for item in os.listdir(new_eval_dir):
                src = os.path.join(new_eval_dir, item)
                dst = os.path.join(nextcloud_dir, "evaluation", item)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

        scenario_config_dst = os.path.join(
            scenario_dir, "scenario_config.json"
        )
        shutil.copy(scenario_config_file, scenario_config_dst)
        print(f"Moved files to {scenario_dir}.")
        if use_nextcloud:
            scenario_config_nextcloud_dst = os.path.join(
                nextcloud_dir, "scenario_config.json"
            )
            shutil.move(scenario_config_file, scenario_config_nextcloud_dst)
            print(f"Moved files to {nextcloud_dir}.")

        time.sleep(10)


if __name__ == "__main__":
    main()
