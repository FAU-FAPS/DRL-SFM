"""Script for running multiple evaluations across different worlds,
people counts, and planners."""

import subprocess
import os
import time

# Define the worlds, number of people, and planners
worlds = [
    "small_hospital",
    "small_house",
]
people_counts = [
    5,
    10,
    15,
]
planners = [
    #"DWA",
    #"MPPI",
    #"SAC-HuMap-LeNet-Cost2",
    #"SAC-HuMap-LeNet-Cost2_VO",
    "SAC-HuMap-LeNet-Cost2_SFM",
]

num_evaluations = "100"  # Number of evaluations to run - default 100
num_simulations = "1"  # Number of simulations to run - default 4
first_ros_domain_id = "60"  # First ROS domain ID - default 60
use_gzclient = "True"  # Whether to use gzclient or not - default true

# Start the timer
start_time = time.time()

# Iterate over each combination of world, people count, and planner
for world in worlds:

    for people in people_counts:

        for planner in planners:

            # Define the evaluation folder path
            evaluation_folder = f"evaluation/{world}_{people}_{planner}"

            # Check whether the evaluation is already done
            if os.path.exists(evaluation_folder) and len(
                os.listdir(evaluation_folder)
            ) == (int(num_evaluations) + 3):
                print(
                    f"Skipping evaluation for world: {world}, "
                    f"people: {people}, planner: {planner} as it is already "
                    f"done len is {len(os.listdir(evaluation_folder))}."
                )
                continue

            if planner == "DRL-VO_Reconstruction":
                command = [
                    "python3",
                    "start_evaluation.py",
                    "--num_evaluations",
                    num_evaluations,
                    "--num_simulations",
                    "4",
                    "--first_ros_domain_id",
                    first_ros_domain_id,
                    "--world_name",
                    world,
                    "--use_gzclient",
                    use_gzclient,
                    "--num_people",
                    str(people),
                    "--planner",
                    planner,
                    "--fast",
                    "True",
                ]
                print(
                    f"Starting evaluation for world: {world}, use_gzclient: "
                    f"{use_gzclient} people: {people}, planner: {planner}"
                )
            else:
                command = [
                    "python3",
                    "start_evaluation.py",
                    "--num_evaluations",
                    num_evaluations,
                    "--num_simulations",
                    num_simulations,
                    "--first_ros_domain_id",
                    first_ros_domain_id,
                    "--world_name",
                    world,
                    "--use_gzclient",
                    use_gzclient,
                    "--num_people",
                    str(people),
                    "--planner",
                    planner,
                    "--fast",
                    "True",
                ]
                print(
                    f"Starting evaluation for world: {world}, use_gzclient: "
                    f"{use_gzclient} people: {people}, planner: {planner}"
                )

            tried_once = False

            # Check if the folder exists and whether all files exist
            while not os.path.exists(evaluation_folder) or len(
                os.listdir(evaluation_folder)
            ) != (int(num_evaluations) + 3):
                if tried_once:
                    print(
                        f"Number of files in {evaluation_folder}: "
                        f"{len(os.listdir(evaluation_folder))}. Retrying..."
                    )
                else:
                    print(
                        f"Starting evaluation for world: {world}, "
                        f"people: {people}, planner: {planner}"
                    )
                tried_once = True
                process = subprocess.run(command)

            print(
                f"Completed evaluation for world: {world}, "
                f"people: {people}, planner: {planner}"
            )

# Print the total time taken for all evaluations
total_time = time.time() - start_time
print(f"Total time taken for all evaluations: {total_time:.2f} seconds")
