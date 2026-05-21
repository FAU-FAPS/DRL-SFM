"""Gymnasium environment for wheelchair navigation in human environments."""

import rclpy
from gymnasium import Env
from gymnasium.spaces import Dict, Box
import numpy as np
from hunav_rl.robot_controller import RobotController
import math
from nav_msgs.msg import Path
import cv2
import hunav_rl.rewards as rew
from people_msgs.msg import People, Person
np.float = float


class WheelchairEnv(RobotController, Env):
    """Gymnasium environment for wheelchair navigation training with
    reinforcement learning."""

    def __init__(self, env_id=0, observation_mode="costmap", mode="train"):
        """Initialize the wheelchair environment.

        Args:
            env_id: Environment identifier for multi-environment training.
            observation_mode: Type of observation space ('costmap' or 'humap').
            mode: Training mode.
        """
        if not rclpy.ok():
            rclpy.init(args=None)
        super().__init__(env_id=env_id)

        # Environment parameters
        self.ENV_ID = env_id
        self.OBSERVATION_MODE = observation_mode
        self.MODE = mode
        self.CAMERA_FOV = 110  # degrees
        self.COSTMAP_RESOLUTION = 120  # cells
        self.MAX_LINEAR_VELOCITY = 1.0  # m/s
        self.MAX_ANGULAR_VELOCITY = 1.0  # rad/s
        self.MINIMUM_DIST_FROM_TARGET = 0.25  # m
        self.MINIMUM_DIST_FROM_OBSTACLES = 0.22  # m
        self.PATH_LENGTH = 7.0  # m
        self.LOOKAHEAD_DISTANCE = 1.0  # m
        self.NUM_WAYPOINTS = 3

        # Initialize state variables
        self._target_location = np.array([0.0, 0.0], dtype=np.float32)
        self._old_dist_to_wp = 0.0
        self._old_wp = None
        self._step_time = self.get_clock().now().nanoseconds * 1e-9
        self._path_update_time = self.get_clock().now().nanoseconds * 1e-9
        self._max_time = 20.0
        self._min_time = 5.0
        self._previous_step_time = None
        self._collision = False
        self._done_collision = False
        self._reached_target = False
        self._done_target = False
        self._reward_sum = 0.0
        self._received_new_data = False
        self._global_path = Path()
        self._received_global_path = False
        self._whole_path = Path()
        
        # Call initialization methods
        self.load_config()
        self.update_laser_reset_time()
        self.define_action_space()
        self.define_observation_space()

    def define_action_space(self):
        """Define the action space for the environment."""
        self.action_space = Box(
            low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32
        )

    def define_observation_space(self):
        """Define the observation space for the environment."""
        if self.OBSERVATION_MODE == "costmap":
            self.observation_space = Dict(
                {
                    "waypoint_distances": Box(
                        low=np.zeros(self.NUM_WAYPOINTS),
                        high=np.ones(self.NUM_WAYPOINTS),
                        dtype=np.float32,
                    ),
                    "waypoint_directions": Box(
                        low=-np.ones(self.NUM_WAYPOINTS),
                        high=np.ones(self.NUM_WAYPOINTS),
                        dtype=np.float32,
                    ),
                    "costmap": Box(
                        low=0,
                        high=1,
                        shape=(
                            self.COSTMAP_RESOLUTION,
                            self.COSTMAP_RESOLUTION,
                            1,
                        ),
                        dtype=np.float32,
                    ),
                }
            )
        if self.OBSERVATION_MODE == "humap":
            self.observation_space = Dict(
                {
                    "waypoint_distances": Box(
                        low=np.zeros(self.NUM_WAYPOINTS),
                        high=np.ones(self.NUM_WAYPOINTS),
                        dtype=np.float32,
                    ),
                    "waypoint_directions": Box(
                        low=-np.ones(self.NUM_WAYPOINTS),
                        high=np.ones(self.NUM_WAYPOINTS),
                        dtype=np.float32,
                    ),
                    "humap": Box(
                        low=-1,
                        high=1,
                        shape=(
                            self.COSTMAP_RESOLUTION,
                            self.COSTMAP_RESOLUTION,
                            3,
                        ),
                        dtype=np.float32,
                    ),
                }
            )

    def step(self, action):
        """Perform one step in the environment with the given action.

        Args:
            action: The action to be performed.

        Returns:
            observation: The observation after performing the action.
            reward: The reward received after performing the action.
            terminated: Whether the episode has terminated.
            timeout: Whether the episode has timed out.
            info: Additional information about the step.
        """
        action = self.denormalize_action(action)
        self.send_velocity_command(action)
        self.spin()
        self._step_time = self.get_clock().now().nanoseconds * 1e-9
        # update the path every second
        old_path = Path()
        old_path = self._global_path
        if old_path is None or len(old_path.poses) == 0:
            self.get_logger().info("No path received yet")
            old_path = self._whole_path
        if self._step_time - self._path_update_time > 1.0:
            self._path_update_time = self._step_time
            if self.MODE == "train":
                start_pos = [
                    self._agent_location[0],
                    self._agent_location[1],
                    self._agent_orientation,
                ]
                self.get_path(start_pos, self._target_location, old_path)
        info = self._get_info()
        self.waypoints = self.get_floating_waypoints(
            self._global_path, self.LOOKAHEAD_DISTANCE
        )
        observation = self._get_obs()

        timeout = self._step_time >= self._max_time + self._reset_time
        reward = self.compute_rewards(info, timeout)
        terminated = (self._done_target == True) or (
            self._done_collision == True
        )
        self._old_dist_to_wp = math.dist(
            self._agent_location[:2], self.waypoints[0][:2]
        )
        self._old_wp = self.waypoints[0][:2]
        return observation, reward, terminated, timeout, info

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state.

        Args:
            seed: Random seed.
            options: Additional options for resetting the environment.

        Returns:
            observation: The initial observation after resetting
            the environment.
            info: Additional information about the reset.
        """
        self.get_logger().info("Resetting environment " + str(self.ENV_ID))
        self._done_set_rob_state = False
        while not self._done_set_rob_state:
            self.send_velocity_command([0.0, 0.0])
            rclpy.spin_once(self)
            if self.MODE == "train":
                robot_start_pos = self.generate_path()
                self._path_update_time = (
                    self.get_clock().now().nanoseconds * 1e-9
                )
                self.reset_robot(robot_start_pos)
            self._target_location = (
                self._global_path.poses[-1].pose.position.x,
                self._global_path.poses[-1].pose.position.y,
            )
        self.terminal_output("Robot position reset")
        self.waypoints = self.get_floating_waypoints(
            self._global_path, self.LOOKAHEAD_DISTANCE
        )
        robot_orientation = np.arctan2(
            2 * (robot_start_pos[2] * robot_start_pos[3]),
            1 - 2 * (robot_start_pos[2] ** 2 + robot_start_pos[3] ** 2),
        )
        start_rotation = self.get_angle_to_wp(
            robot_start_pos, robot_orientation
        )
        self.terminal_output("Start rotation: " + str(start_rotation))
        self._min_time = (
            abs(start_rotation) / self.MAX_ANGULAR_VELOCITY
            + self.PATH_LENGTH / self.MAX_LINEAR_VELOCITY
        )
        self._start_rotation = start_rotation
        self.terminal_output("Minimum time: " + str(self._min_time))
        self._max_time = self._min_time * 4
        self._old_dist_to_wp = math.dist(
            robot_start_pos[:2], self.waypoints[0][:2]
        )
        self._old_wp = self.waypoints[0][:2]
        self._whole_path = self._global_path
        self._done_reset_rtabmap = False
        self.reset_rtabmap()
        while not self._done_reset_rtabmap:
            self.send_velocity_command([0.0, 0.0])
            rclpy.spin_once(self)
        self._received_new_data = False
        self.send_velocity_command([0.0, 0.0])
        self.update_laser_reset_time()
        self.spin()
        self._collision = False
        self._done_collision = False
        self._reached_target = False
        self._done_target = False
        observation = self._get_obs()
        info = self._get_info()
        self._old_dist = info["distance"]
        self._reset_time = self.get_clock().now().nanoseconds * 1e-9
        self._step_time = self.get_clock().now().nanoseconds * 1e-9

        if self._use_sfm_prediction:
            self._previous_step_time = None
        self._reward_sum = 0.0
        return observation, info

    def _get_obs(self):
        """Get the current observation from the environment.

        Returns:
            obs: The current observation.
        """
        local_costmap_rotated = None
        if (
            self.OBSERVATION_MODE == "costmap"
            or self.OBSERVATION_MODE == "humap"
        ):
            local_costmap_rotated = self._rotate_costmap()
            if self.OBSERVATION_MODE == "humap":
                human_vel_map = self._get_human_vel_map()
                humap = np.concatenate(
                    (local_costmap_rotated, human_vel_map), axis=2
                )
        waypoint_distances, waypoint_directions = (
            self._get_waypoints_in_polar_coordinates()
        )
        if self.OBSERVATION_MODE == "costmap":
            obs = {
                "waypoint_distances": waypoint_distances,
                "waypoint_directions": waypoint_directions,
                "costmap": local_costmap_rotated,
            }
        elif self.OBSERVATION_MODE == "humap":
            obs = {
                "waypoint_distances": waypoint_distances,
                "waypoint_directions": waypoint_directions,
                "humap": humap,
            }

        obs = self.normalize_observation(obs)
        self._waypoint_distances = obs["waypoint_distances"]
        self._waypoint_directions = obs["waypoint_directions"]
        self.publish_markers()
        return obs

    def _get_info(self):
        """Get additional information about the current state of the
        environment.

        Returns:
            info: A dictionary containing additional information.
        """
        info = {
            "distance": math.dist(
                self._agent_location[:2], self._target_location[:2]
            ),
            "progress_distance": math.dist(
                self._agent_location[:2], self._old_wp[:2]
            ),
        }
        return info

    def spin(self):
        """Spin the environment, processing callbacks and updating state."""
        self._done_pose = False
        self._done_laser = False
        while not (
            self._done_pose and self._done_laser and self._received_new_data
        ):
            rclpy.spin_once(self)

    def generate_path(self):
        """Generate a new path for the robot to follow.

        Returns:
            robot_start_pos: The starting position of the robot on the new path.
        """
        self.waypoints = []
        is_long_enough = False
        while not is_long_enough:
            self._global_path = Path()
            robot_start_pos = self.randomize_robot_location()
            target_pos = self.randomize_target_location()
            self.get_path(robot_start_pos, target_pos)
            is_long_enough = self.is_long_enough(
                self._global_path, self.PATH_LENGTH
            )
        self.waypoints = self.get_floating_waypoints(
            self._global_path, self.LOOKAHEAD_DISTANCE
        )
        return robot_start_pos

    def get_angle_to_wp(self, robot_start_pos, robot_start_rot):
        """Calculate the angle to the next waypoint.

        Args:
            robot_start_pos: The starting position of the robot.
            robot_start_rot: The starting rotation of the robot.

        Returns:
            angle_diff: The difference in angle to the next waypoint.
        """
        if len(self.waypoints) > 0:
            first_waypoint = self.waypoints[0]
            x_diff = first_waypoint[0] - robot_start_pos[0]
            y_diff = first_waypoint[1] - robot_start_pos[1]
            angle = np.arctan2(y_diff, x_diff)
            angle_diff = angle - robot_start_rot
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            return angle_diff

    def reset_robot(self, robot_start_pos):
        """Reset the robot's state.

        Args:
            robot_start_pos: The starting position of the robot.
        """
        self.call_set_robot_state_service(robot_start_pos)
        iteration = 0
        max_iterations = 1000
        while not self._done_set_rob_state:
            self.send_velocity_command([0.0, 0.0])
            rclpy.spin_once(self)
            if not self._reset_possible:
                if self._human_too_close:
                    self.get_logger().info(
                        "Robot too close to human, trying other position..."
                    )
                    break
                self.terminal_output("Service not responding, retrying...")
                self.call_set_robot_state_service(robot_start_pos)
            if iteration >= max_iterations:
                break
            iteration += 1

    def _rotate_costmap(self):
        """Rotate the costmap to align with the robot's orientation.

        Returns:
            local_costmap_rotated: The rotated costmap.
        """
        local_costmap_rotated = None
        angle_deg = -np.degrees(self._agent_orientation)
        costmap = np.fliplr(self._local_costmap)
        costmap_2d = np.ascontiguousarray(costmap[:, :, 0])
        costmap_2d = costmap_2d.astype(np.float32)
        (h, w) = costmap_2d.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated_costmap = cv2.warpAffine(
            costmap_2d,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        rotated_costmap = rotated_costmap[..., np.newaxis]
        crop_h = int(h * 6.0 / 8.0)
        crop_w = int(w * 6.0 / 8.0)
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        cropped_costmap = rotated_costmap[
            start_y : start_y + crop_h, start_x : start_x + crop_w, :
        ]
        reshaped_costmap = cv2.resize(
            cropped_costmap, (120, 120), interpolation=cv2.INTER_LINEAR
        )
        reshaped_costmap = np.rot90(reshaped_costmap, k=3)
        local_costmap_rotated = reshaped_costmap[..., None]
        return local_costmap_rotated

    def _get_human_vel_map(self):
        """Get the human velocity map.

        Returns:
            human_vel_map: The human velocity map.
        """
        human_vel_map = np.zeros(
            (self.COSTMAP_RESOLUTION, self.COSTMAP_RESOLUTION, 2),
            dtype=np.float32,
        )
        if len(self._agents.people) < 1:
            return human_vel_map
        agent_mapframe_positions = np.array(
            [
                [agent.position.x, agent.position.y]
                for agent in self._agents.people
            ]
        )
        agent_mapframe_velocities = np.array(
            [
                [agent.velocity.x, agent.velocity.y]
                for agent in self._agents.people
            ]
        )
        # Create a rotation matrix to transform from global to robot frame:
        R_pos = np.array(
            [
                [
                    np.cos(self._agent_orientation),
                    np.sin(self._agent_orientation),
                ],
                [
                    -np.sin(self._agent_orientation),
                    np.cos(self._agent_orientation),
                ],
            ]
        )
        # Transform positions and velocities using the rotation matrix.
        robot_coords = (
            agent_mapframe_positions - self._agent_location
        ) @ R_pos.T
        agent_robotframe_velocities = agent_mapframe_velocities @ R_pos.T
        robot_target_x, robot_target_y = robot_coords[:, 0], robot_coords[:, 1]
        agent_robotframe_vel_x = agent_robotframe_velocities[:, 0]
        agent_robotframe_vel_y = agent_robotframe_velocities[:, 1]
        # Only consider points in front of the robot: from -55 to +55 degrees.
        angles_deg = np.degrees(np.arctan2(robot_target_y, robot_target_x))
        front_mask = (angles_deg >= -self.CAMERA_FOV / 2) & (
            angles_deg <= self.CAMERA_FOV / 2
        )
        # Calculate positions in the map.
        raw_x_map = (
            robot_target_x
            + (6 / self.COSTMAP_RESOLUTION) * (self.COSTMAP_RESOLUTION / 2)
        ) / (6 / self.COSTMAP_RESOLUTION)
        raw_y_map = (
            robot_target_y
            + (6 / self.COSTMAP_RESOLUTION) * (self.COSTMAP_RESOLUTION / 2)
        ) / (6 / self.COSTMAP_RESOLUTION)
        x_map = (self.COSTMAP_RESOLUTION - raw_x_map).astype(int)
        y_map = (self.COSTMAP_RESOLUTION - raw_y_map).astype(int)
        # Filter valid positions inside the map bounds.
        valid_mask = (
            (x_map >= 0)
            & (x_map < self.COSTMAP_RESOLUTION)
            & (y_map >= 0)
            & (y_map < self.COSTMAP_RESOLUTION)
        )
        combined_mask = front_mask & valid_mask
        x_map = x_map[combined_mask]
        y_map = y_map[combined_mask]
        agent_robotframe_vel_x = agent_robotframe_vel_x[combined_mask]
        agent_robotframe_vel_y = agent_robotframe_vel_y[combined_mask]
        # Store the transformed velocities into the human velocity map.
        human_vel_map[x_map, y_map, 0] = agent_robotframe_vel_x
        human_vel_map[x_map, y_map, 1] = agent_robotframe_vel_y
        self._human_vel_map = human_vel_map
        return human_vel_map

    def _get_waypoints_in_polar_coordinates(self):
        """Get the waypoints in polar coordinates (radius, angle).

        Returns:
            waypoint_distances: The radius of the waypoints.
            waypoint_directions: The angle of the waypoints.
        """
        waypoint_distances = []
        waypoint_directions = []
        for i in range(self.NUM_WAYPOINTS):
            waypoint = self.waypoints[i]
            # Compute radius: distance from the agent to the waypoint
            radius = math.dist(self._agent_location[:2], waypoint[:2])
            # Transform the waypoint coordinates into the robot’s frame
            robot_target_x = math.cos(-self._agent_orientation) * (
                waypoint[0] - self._agent_location[0]
            ) - math.sin(-self._agent_orientation) * (
                waypoint[1] - self._agent_location[1]
            )
            robot_target_y = math.sin(-self._agent_orientation) * (
                waypoint[0] - self._agent_location[0]
            ) + math.cos(-self._agent_orientation) * (
                waypoint[1] - self._agent_location[1]
            )
            theta = math.atan2(robot_target_y, robot_target_x)
            waypoint_distances.append(radius)
            waypoint_directions.append(theta)
        return np.array(waypoint_distances, dtype=np.float32), np.array(
            waypoint_directions, dtype=np.float32
        )

    def randomize_target_location(self):
        """Randomly generate a target location.

        Returns:
            target_location: The randomly generated target location.
        """
        x, y = self.get_random_spawn_position()
        target_location = [x, y]
        return target_location

    def randomize_robot_location(self):
        """Randomly generate a robot location.

        Returns:
            robot_start_pos: The randomly generated robot location.
        """
        position_x, position_y = self.get_random_spawn_position()
        angle = float(math.radians(np.random.uniform(-180, 180)))
        orientation_z = float(math.sin(angle / 2))
        orientation_w = float(math.cos(angle / 2))
        return [position_x, position_y, orientation_z, orientation_w]

    def compute_rewards(self, info, timeout=False):
        """Compute the rewards for the current step.

        Args:
            info: Information about the current state.
            action: The action taken in the current step.
            observation: The observation received in the current step.
            timeout: Whether the episode has timed out.

        Returns:
            reward: The computed reward.
        """
        rewards = []

        # Goal reward
        goal_reached = self._reached_target
        t_max = self._max_time
        t_min = self._min_time
        t = self._step_time - self._reset_time
        goal_reward = rew.goal_reward(goal_reached, t_max, t_min, t)
        if goal_reached:
            self.terminal_output("TARGET REACHED")
            self._done_target = True
        rewards.append(goal_reward)

        # Truncated Reward
        truncated = self._collision or timeout
        l_max = self.PATH_LENGTH
        if truncated:
            positions = np.array(
                [
                    [p.pose.position.x, p.pose.position.y]
                    for p in self._global_path.poses
                ]
            )
            robot_pos = self._agent_location[0:2]
            dists = np.linalg.norm(positions - robot_pos, axis=1)
            closest_idx = np.argmin(dists)
            positions = positions[closest_idx:]
            segs = positions[1:] - positions[:-1]
            seg_lengths = np.linalg.norm(segs, axis=1)
            l_t = seg_lengths.sum()
        else:
            l_t = 0.0
        truncation_reward = rew.truncation_reward(truncated, l_t, l_max)
        if self._collision:
            self.terminal_output("HIT AN OBSTACLE.")
            self._done_collision = True
        if timeout:
            self.terminal_output("TIMEOUT.")
        rewards.append(truncation_reward)

        # Progress reward
        if self._old_wp is None:
            progress_rew = 0.0
        else:
            d_prog = self._old_dist_to_wp - info["progress_distance"]
            progress_rew = rew.progress_reward(d_prog)
        rewards.append(progress_rew)

        # Heading reward
        angle_tol = np.pi/9
        if self._use_angle_reward:
            alpha_wp = self.get_angle_to_wp(
                self._agent_location, self._agent_orientation
            )
            heading_reward = rew.heading_reward(alpha_wp, angle_tol=angle_tol)
            rewards.append(heading_reward)

        # Cost reward
        if self._use_cost_as_reward:
            if self._local_costmap is not None:
                cost_reward = rew.cost_reward(self._local_costmap)
            else:
                cost_reward = 0.0
                self.terminal_warning("No local costmap received")
            rewards.append(cost_reward)

        # SFM reward
        if self._use_sfm_prediction:
            if self._previous_step_time is None:
                self.compute_sfm_force(-1.0)
                sfm_reward = 0.0
            else:
                dt = self._step_time - self._previous_step_time
                if dt < 0.0 or dt > 1.0:
                    self.compute_sfm_force(-1.0)
                    self.terminal_warning(
                        "Time between steps was too low or too high:"
                    )
                    sfm_reward = 0.0
                else:
                    prediction = self.compute_sfm_force(dt)
                    if prediction is None:
                        self.terminal_warning("SFM prediction failed")
                        sfm_reward = 0.0
                    else:
                        sfm_reward = rew.sfm_reward(
                            self._agent_location[:2], prediction, dt
                        )
            self._previous_step_time = self._step_time
            rewards.append(sfm_reward)

        # VO reward
        if self._use_vo_reward:
            # Transform the target location to the robot's frame
            delta_global = self._target_location[:2] - self._agent_location[:2]
            theta = self._agent_orientation
            cos_theta = np.cos(-theta)
            sin_theta = np.sin(-theta)
            rotation_matrix = np.array(
                [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
            )
            relative_goal = rotation_matrix @ delta_global
            # Transform pedestrians to the robot's frame
            relative_agents = People()
            for agent in self._agents.people:
                relative_agent = Person()
                agent_pos = np.array([agent.position.x, agent.position.y])
                relative_agent_pos = rotation_matrix @ (
                    agent_pos - self._agent_location[:2]
                )
                relative_agent.position.x = relative_agent_pos[0]
                relative_agent.position.y = relative_agent_pos[1]
                agent_vel = np.array([agent.velocity.x, agent.velocity.y])
                relative_agent_vel = rotation_matrix @ agent_vel
                relative_agent.velocity.x = relative_agent_vel[0]
                relative_agent.velocity.y = relative_agent_vel[1]
                relative_agents.people.append(relative_agent)
            vo_reward = rew.vo_reward(
                relative_goal, relative_agents, self._agent_lin_vel[0]
            )
            if vo_reward is None:
                vo_reward = 0.0
                self.terminal_warning(
                    "Velocity obstacle reward could not be computed"
                )
            rewards.append(vo_reward)

        # Sum all reward components
        reward = sum(rewards)
        # No positive reward if linear velocity < 0.2 to avoid standing still
        if self._agent_lin_vel[0] < 0.2:
            self.terminal_output("Standing still")
            if (reward > 0.0) and (self._reached_target == False):
                    reward = 0.0
        if self._use_angle_reward:
            if alpha_wp <= angle_tol:
                if self._start_rotation > angle_tol:
                    reward += self._start_rotation/angle_tol
                    self._start_rotation = 0.0
        self._reward_sum += reward
        rewards.append(float(reward))
        rewards.append(float(self._reward_sum))
        self.send_reward(rewards)
        return reward

    def normalize_observation(self, observation):
        """Normalize the observation values to a standard range.

        Args:
            observation: The original observation.

        Returns:
            observation: The normalized observation.
        """
        # Normalize distances to [0,1]
        observation["waypoint_distances"] = (
            (
                observation["waypoint_distances"]
                / (self.NUM_WAYPOINTS * self.LOOKAHEAD_DISTANCE)
            )
            .astype(np.float32)
        )
        # Normalize angles from [-pi, pi] to [-1,1]
        observation["waypoint_directions"] = (
            (observation["waypoint_directions"] / math.pi)
            .astype(np.float32)
        )

        if self.OBSERVATION_MODE == "costmap":
            observation["costmap"] = (
                (observation["costmap"] / 100).astype(np.float32)
            )
            self.publish_costmap_as_img(observation["costmap"])
        if self.OBSERVATION_MODE == "humap":
            humap_p = observation["humap"].astype(
                np.float32
            )  # humap_p is for visualization only
            humap = observation["humap"].astype(np.float32)
            humap_p[..., 0] = humap_p[..., 0] / 100
            humap[..., 0] = humap[..., 0] / 100
            humap_p[..., 1:] = (humap_p[..., 1:] + 3) / 6
            publish_humap = humap_p
            # For the human velocity channels scale to [-1,1]
            humap[..., 1:] = (humap[..., 1:]) / 3
            observation["humap"] = humap
            self.publish_costmap_as_img(publish_humap)
        return observation

    def denormalize_action(self, normalized_action):
        """Denormalize the action values to the original range.

        Args:
            normalized_action: The normalized action.

        Returns:
            action: The denormalized action.
        """
        action_linear = (
            (self.MAX_LINEAR_VELOCITY * (normalized_action[0] + 1))
        ) / 2
        action_angular = (
            (self.MAX_ANGULAR_VELOCITY * (normalized_action[1] + 1))
            + (-self.MAX_ANGULAR_VELOCITY * (1 - normalized_action[1]))
        ) / 2
        return np.array([action_linear, action_angular], dtype=np.float32)

    def close(self):
        """Clean up and close the environment."""
        self.destroy_node()
        rclpy.shutdown()
