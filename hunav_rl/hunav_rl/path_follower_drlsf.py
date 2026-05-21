"""Path follower node for DRL-SFM."""


import rclpy
from gymnasium import Env
import numpy as np
import math
import os
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
import numpy as np
np.float = float
from nav_msgs.msg import OccupancyGrid
from nav2_rl_controller_msgs.action import CalcTwist
import cv2
from geometry_msgs.msg import Twist
import threading
from people_msgs.msg import People
from stable_baselines3 import SAC, PPO
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
import time
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from rclpy.duration import Duration
from rclpy.clock import JumpThreshold
from ament_index_python import get_package_share_directory


class NavEnv(Node,Env):
    """Path follower using deep reinforcement learning with social forces."""
    def __init__(self, observation_mode="humap", planner_model="best_model"):
        """
        Initialize the path follower environment node.

        Args:
            observation_mode (str): Type of observation space
            ('costmap', 'humap').
            planner_model (str): Name of the planner model to load.
        """
        super().__init__('path_follower')
        self.declare_parameter('observation_mode', "humap")
        self.declare_parameter('planner_model', "best_model")
        self._observation_mode = (
            self.get_parameter('observation_mode')
            .get_parameter_value().string_value
        )
        self._planner_model = self.get_parameter(
            'planner_model'
        ).get_parameter_value().string_value
        parts = self._planner_model.split('-')
        if len(parts) < 4:
            raise ValueError(
                f"Spec '{self._planner_model}' must have at least 4 '-' " 
                f"separated parts."
            )
        self.drl_alg, self.observation_mode_temo, self.architecture, \
            self.reward_function = parts[:4]

        # Environment parameters
        self.NUM_WAYPOINTS = 3
        self.GOAL_RADIUS = 0.3
        self.MAX_LINEAR_VELOCITY = 1.0
        self.ANGULAR_VELOCITY = 1.0
        self.COSTMAP_RESOLUTION = 120
        self.CAMERA_FOV = 110.0
        self.MAP_FRAME = "map"
        self.ROBOT_FRAME = "base_footprint"

        # Initialize state variables
        self._agent_location = np.zeros(2, dtype=np.float32)
        self._agent_orientation = 0.0
        self._agents = None
        self._local_costmap = None
        self._laser_reads = None
        self._human_vel_map = None
        self._done_pose = False
        self._done_local_costmap = False
        self._done_laser = False
        self._waypoint_distances = np.zeros(
            self.NUM_WAYPOINTS, dtype=np.float32
        )
        self._waypoint_directions = np.zeros(
            self.NUM_WAYPOINTS, dtype=np.float32
        )
        self._goal_lock = threading.Lock()

        # Subscribers
        self.agents_sub = self.create_subscription(
            People, 'people', self.human_states_callback, 1
        )
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid, '/local_costmap/costmap',
            self.local_costmap_callback, 1
        )
        
        # Publishers
        self.marker_pub = self.create_publisher(
            Marker, 'observation', 10
        )
        self.image_pub_costmap = self.create_publisher(
            Image, 'costmap_image', 10
        )
        self.pub_cmd = self.create_publisher(
            Twist, "/cmd_vel", 1
        )
        self.pub_path = self.create_publisher(
            Path, "/action_path", 1
        )

        # TF2 Initialization
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        time.sleep(1.0)  # Allow time for TF listener to initialize
        # Reset TF buffer on time jumps
        self.timer = None
        threshold = JumpThreshold(
            min_forward=None,
            min_backward=Duration(seconds=-0.1),
            on_clock_change=True
        )
        self.jump_handle = self.get_clock().create_jump_callback(
            threshold, post_callback=self.on_time_jump
        )

        # Load DRL model
        pkg_share_dir = get_package_share_directory('hunav_rl')
        ws_dir = os.path.abspath(
            os.path.join(pkg_share_dir, '..', '..', '..', '..')
        )
        model_path = os.path.join(
            ws_dir,
            'src/drl-sfm/hunav_rl/hunav_rl/rl_models/',
            self._planner_model
        )
        self.get_logger().info(f"Loading model from {model_path}")
        if self.drl_alg == "SAC":
            self.get_logger().info("Loading SAC!")
            self.model = SAC.load(
                model_path, policy_only=True, device='auto'
            )
        elif self.drl_alg == "PPO":
            self.get_logger().info("Loading PPO")
            self.model = PPO.load(
                model_path, policy_only=True, device='auto'
            )
        else:
            self.get_logger().error(
                f"Unsupported DRL algorithm: {self.drl_alg}"
            )
            raise ValueError(
                f"Unsupported DRL algorithm: {self.drl_alg}"
            )

    def on_time_jump(self, event):
        """
        Handle time jumps detected by the ROS 2 clock.

        Args:
            event: The time jump event.
        """
        self.get_logger().info("Time jump detected - clearing TF buffer")
        self.tf_buffer.clear()
        self.tf_listener.unregister()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def human_states_callback(self, msg: People):
        """
        Callback for updating human states (positions and velocities).

        Args:
            msg (People): Message containing human agent states.
        """
        self._agents = msg

        if not self._agents or not self._agents.people:
            return
        
        transform_robot_map = None
        try:
            transform_robot_map = self.tf_buffer.lookup_transform(
                self.ROBOT_FRAME,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
            
        except TransformException as ex:
            self.get_logger().error(
                f"Failed to transform agent position: {ex}"
            )
            return
        
        # Transform human positions to the robot frame
        for agent in self._agents.people:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = msg.header.frame_id
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x = agent.position.x
            point_stamped.point.y = agent.position.y
            point_stamped.point.z = agent.position.z
            transformed_point = do_transform_point(
                point_stamped, transform_robot_map
            )
            agent.position.x = transformed_point.point.x
            agent.position.y = transformed_point.point.y
            agent.position.z = transformed_point.point.z
            rotation = transform_robot_map.transform.rotation
            qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
            rotation_matrix = np.array([
                [1 - 2 * (qy**2 + qz**2),
                 2 * (qx * qy - qz * qw),
                 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw),
                 1 - 2 * (qx**2 + qz**2),
                 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw),
                 2 * (qy * qz + qx * qw),
                 1 - 2 * (qx**2 + qy**2)]
            ])
            velocity_global = np.array([
                agent.velocity.x, agent.velocity.y, 0.0
            ])
            velocity_robot = rotation_matrix @ velocity_global
            agent.velocity.x = velocity_robot[0]
            agent.velocity.y = velocity_robot[1]
    
    def local_costmap_callback(self, msg: OccupancyGrid):
        """
        Callback for updating the local costmap.

        Args:
            msg (OccupancyGrid): Message containing costmap data.
        """
        self._local_costmap = np.array(msg.data).reshape(
            (msg.info.height, msg.info.width, 1)
        )
        self._done_local_costmap = True

    def publish_target_location(self):
        """
        Publish the target location as a marker.
        """
        goal_pose = self._target_location
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        points = [Point(x=goal_pose[0], y=goal_pose[1], z=0.1)]
        marker.points = points
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.r = 151/255
        marker.color.g = 193/255
        marker.color.b = 57/255
        self.marker_pub.publish(marker)

    def publish_humans_location(self):
        """
        Publish the locations of detected humans as markers.
        """
        marker = Marker()
        marker.header.frame_id = self.ROBOT_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "humans"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        points = []
        for agent in self._agents.people:
            pt = Point(x=agent.position.x, y=agent.position.y, z=0.1)
            points.append(pt)
        marker.points = points
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.r = 255/255
        marker.color.g = 203/255
        marker.color.b = 0/255
        self.marker_pub.publish(marker)

    def publish_robot_location(self):
        """
        Publish the robot's location as a marker.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(self._agent_location[0])
        marker.pose.position.y = float(self._agent_location[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.45
        marker.scale.y = 0.45
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        self.marker_pub.publish(marker)

    def publish_obs(self):
        """
        Publish the agent's observations as markers.
        """
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "agent_obs"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        points = []
        radius = self._waypoint_distances
        angle = self._waypoint_directions
        if radius is None or angle is None:
            self.terminal_warning("No observation to publish")
            return
        for i in range(0, len(radius)):
            r_val = radius[i]
            r_val = r_val * self.NUM_WAYPOINTS
            theta = angle[i]
            theta = theta * math.pi
            x = r_val * math.cos(theta)
            y = r_val * math.sin(theta)
            pt = Point(x=x, y=y, z=0.1)
            points.append(pt)
        marker.points = points
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 67/255
        marker.color.b = 89/255
        self.marker_pub.publish(marker)

    def publish_costmap_as_img(self, costmap):
        """
        Publish the costmap as an image.

        Args:
            costmap (np.ndarray): Costmap array to publish.
        """
        if costmap.ndim == 2 or (costmap.ndim == 3 and costmap.shape[-1] == 1):
            costmap_rgb = np.stack((costmap, costmap, costmap), axis=-1)
        else:
            costmap_rgb = costmap
        costmap_rgb = ((1.0 - costmap_rgb) * 255).astype(np.uint8)
        # Mark the 4 pixels in the middle of the image in red (robot)
        center_x, center_y = costmap.shape[1] // 2, costmap.shape[0] // 2
        costmap_rgb[center_y-1:center_y+1, center_x-1:center_x+1] = [255, 0, 0]
        img = Image()
        img.header.frame_id = "map"
        img.header.stamp = self.get_clock().now().to_msg()
        img.height = costmap_rgb.shape[0]
        img.width = costmap_rgb.shape[1]
        img.encoding = 'rgb8'
        img.is_bigendian = 0
        img.step = img.width * 3
        img.data = costmap_rgb.tobytes()
        self.image_pub_costmap.publish(img)

    def publish_map_as_img(self, observation):
        """
        Publish the map (costmap or humap) as an image.
        Only for visualization purposes.

        Args:
            observation (dict): Observation dictionary containing map data.
        """
        if self._observation_mode == "costmap":
            observation["costmap"] = (
                observation["costmap"] / 100
            ).astype(np.float32).round(2)
            self.publish_costmap_as_img(observation["costmap"])
        if self._observation_mode == "humap":
            # humap_p is for visualization only!
            humap_p = observation["humap"].astype(np.float32)
            humap_p[..., 0] = (humap_p[..., 0])
            humap_p[..., 1:] = ((humap_p[..., 1:] + 3) / 6)
            publish_humap = humap_p.round(2).clip(0, 1)
            self.publish_costmap_as_img(publish_humap)

    def publish_markers(self):
        """
        Publish all relevant markers (target, robot, humans, observations).
        """
        self.publish_target_location()
        self.publish_obs()
        self.publish_humans_location()

    def get_robot_location(self):
        """
        Get the robot's location and orientation from the TF tree.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.MAP_FRAME,
                self.ROBOT_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2) 
            )
            self._agent_location= np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ], dtype=np.float32)
            self._agent_orientation = 2 * math.atan2(
                transform.transform.rotation.z,
                transform.transform.rotation.w
            )
        except TransformException as ex:
            self.get_logger().error(f'Failed to get robot location: {ex}')

    def get_obs(self):
        """
        Get the current observation for the agent, including costmap or humap.

        Returns:
            dict: The current observation.
        """
        local_costmap_rotated = None
        human_vel_map = self._get_human_vel_map()
        if (self._observation_mode == "costmap" or
            self._observation_mode == "humap"):
            local_costmap_rotated = self._rotate_costmap()
            if self._observation_mode == "humap":
                humap = np.concatenate(
                    (local_costmap_rotated, human_vel_map),
                    axis=2
                )
        waypoint_distances, waypoint_directions = self._get_polar_coordinates()
        if self._observation_mode == "costmap":
            obs = {
                "waypoint_distances": waypoint_distances,
                "waypoint_directions": waypoint_directions,
                "costmap": local_costmap_rotated
            }
        elif self._observation_mode == "humap":
            obs = {
                "waypoint_distances": waypoint_distances,
                "waypoint_directions": waypoint_directions,
                "humap": humap
            }
        obs = self.normalize_observation(obs)
        self._waypoint_distances = obs["waypoint_distances"]
        self._waypoint_directions = obs["waypoint_directions"]
        return obs

    def get_floating_waypoints(self, path, lookahead_distance=1.0):
        """
        Calculates floating waypoints on the provided path that are the given
        lookahead_distance ahead of the robot.
        
        Args:
            path (Path): The path to follow.
            lookahead_distance (float): Distance ahead for waypoints.
        Returns:
            list: List of waypoints (x, y).
        """
        num_waypoints = self.NUM_WAYPOINTS
        waypoints = []
        if not path.poses:
            self.get_logger().error("Empty path received!")
            return (0, 0), (0, 0, 0)
        poses = np.array([
            [p.pose.position.x, p.pose.position.y] for p in path.poses
        ])
        robot_pos = self._agent_location[0:2]
        for i in range(num_waypoints):
            dists = np.linalg.norm(poses - robot_pos, axis=1)
            closest_idx = np.argmin(dists)
            sub_poses = poses[closest_idx:]
            if len(sub_poses) < 2:
                goal = sub_poses[-1]
            else:
                segs = sub_poses[1:] - sub_poses[:-1]
                seg_dists = np.linalg.norm(segs, axis=1)
                cum_dists = np.concatenate(([0], np.cumsum(seg_dists)))
                if cum_dists[-1] < lookahead_distance:
                    goal = sub_poses[-1]
                else:
                    idx = np.searchsorted(cum_dists, lookahead_distance)
                    prev_dist = cum_dists[idx - 1]
                    seg_length = cum_dists[idx] - prev_dist
                    ratio = ((lookahead_distance - prev_dist) / seg_length
                             if seg_length else 0)
                    goal = (sub_poses[idx - 1] + ratio *
                            (sub_poses[idx] - sub_poses[idx - 1]))
            waypoints.append((goal[0], goal[1]))
            robot_pos = goal
        self.waypoints = waypoints
        if len(waypoints) < num_waypoints:
            self.get_logger().error("Not enough waypoints found!")
            self.get_floating_waypoints(
                path,
                lookahead_distance=lookahead_distance
            )

    def get_floating_waypoints_adaptive(self, path, window_size=20,
                                        threshold=np.pi/180*5,
                                        lookahead_range=[0.7, 5.0]):
        """
        Calculates adaptive floating waypoints based on path curvature
        and distance.

        Args:
            path (Path): The path to follow.
            window_size (int): Window size for curvature analysis.
            threshold (float): Angle threshold for curvature.
            lookahead_range (list): Min/max lookahead distances.
        Returns:
            list: List of waypoints (x, y).
        """
        num_waypoints = self.NUM_WAYPOINTS
        waypoints = []
        window_size = int(window_size)
        if not path.poses:
            self.get_logger().error("Empty path received!")
            return (0, 0), (0, 0, 0)
        poses = np.array([
            [p.pose.position.x, p.pose.position.y] for p in path.poses
        ])
        robot_pos = self._agent_location[0:2]
        for i in range(num_waypoints):
            dists = np.linalg.norm(poses - robot_pos, axis=1)
            closest_idx = np.argmin(dists)
            sub_poses = poses[closest_idx:]
            if len(sub_poses) <= 1:
                goal = sub_poses[-1]
            else:
                abs_dist = np.linalg.norm(sub_poses - sub_poses[0], axis=1)
                if np.any(abs_dist >= lookahead_range[0]):
                    min_idx = np.argmax(abs_dist >= lookahead_range[0])
                    if np.any(abs_dist >= lookahead_range[1]):
                        max_idx = np.argmax(abs_dist >= lookahead_range[1])
                    else:
                        max_idx = len(abs_dist) - 1
                    angles = np.arctan2(
                        sub_poses[min_idx:, 1] - sub_poses[0][1],
                        sub_poses[min_idx:, 0] - sub_poses[0][0]
                    )
                    angle_diffs = angles - angles[0]
                    if np.any(np.abs(angle_diffs) >= threshold):
                        lookahead_idx = (
                            np.argmax(np.abs(angle_diffs) >= threshold)
                            + min_idx
                        )
                    else:
                        lookahead_idx = max_idx
                    idx = max(min(lookahead_idx, max_idx), min_idx)
                    goal = sub_poses[idx]
                else:
                    goal = sub_poses[-1]
            waypoints.append((goal[0], goal[1]))
            robot_pos = goal
        self.waypoints = waypoints
        if len(waypoints) < num_waypoints:
            self.get_logger().error("Not enough waypoints found!")
            self.get_floating_waypoints_adaptive(
                path, threshold=threshold, lookahead_range=lookahead_range
            )
        return waypoints

    def _get_human_vel_map(self):
        """
        Create a human velocity map based on detected human positions and 
        velocities.

        Returns:
            np.ndarray: Human velocity map.
        """
        self.get_logger().info(
            f'Creating human velocity map with resolution '
            f'{self.COSTMAP_RESOLUTION} and camera FOV {self.CAMERA_FOV}'
        )
        human_vel_map = np.zeros((self.COSTMAP_RESOLUTION,
                      self.COSTMAP_RESOLUTION, 2),
                     dtype=np.float32)
        if not self._agents.people:
            return human_vel_map

        agent_robotframe_positions = np.array(
            [[agent.position.x, agent.position.y]
             for agent in self._agents.people]
        )
        agent_robotframe_velocities = np.array([
            [agent.velocity.x, agent.velocity.y]
            for agent in self._agents.people
        ])
        robot_target_x, robot_target_y = (
            agent_robotframe_positions[:, 0],
            agent_robotframe_positions[:, 1],
        )
        agent_robotframe_vel_x, agent_robotframe_vel_y = (
            agent_robotframe_velocities[:, 0],
            agent_robotframe_velocities[:, 1],
        )
        # Only consider points in front of the robot: from -55 to +55 degrees.
        angles_deg = np.degrees(np.arctan2(robot_target_y, robot_target_x))
        front_mask = (
            (angles_deg >= -self.CAMERA_FOV / 2)
            & (angles_deg <= self.CAMERA_FOV / 2)
        )
        # Calculate positions in the map.
        raw_x_map = (
            (robot_target_x + (6 / self.COSTMAP_RESOLUTION)
             * (self.COSTMAP_RESOLUTION / 2))
            / (6 / self.COSTMAP_RESOLUTION)
        )
        raw_y_map = (
            (robot_target_y + (6 / self.COSTMAP_RESOLUTION)
             * (self.COSTMAP_RESOLUTION / 2))
            / (6 / self.COSTMAP_RESOLUTION)
        )
        x_map = (self.COSTMAP_RESOLUTION - raw_x_map).astype(int)
        y_map = (self.COSTMAP_RESOLUTION - raw_y_map).astype(int)
        # Filter valid positions inside the map bounds.
        valid_mask = (
            (x_map >= 0) & (x_map < self.COSTMAP_RESOLUTION)
            & (y_map >= 0) & (y_map < self.COSTMAP_RESOLUTION)
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

    def _get_polar_coordinates(self):
        """
        Compute the polar coordinates (radius, angle) of the waypoints relative
        to the robot.
        
        Returns:
            tuple: Arrays of waypoint distances and directions.
        """
        waypoint_distances = []
        waypoint_directions = []
        for i in range(len(self.waypoints)):
            waypoint = self.waypoints[i]
            radius = math.dist(self._agent_location[:2], waypoint[:2])
            robot_target_x = (
                math.cos(-self._agent_orientation)
                * (waypoint[0] - self._agent_location[0])
                - math.sin(-self._agent_orientation)
                * (waypoint[1] - self._agent_location[1])
            )
            robot_target_y = (
                math.sin(-self._agent_orientation)
                * (waypoint[0] - self._agent_location[0])
                + math.cos(-self._agent_orientation)
                * (waypoint[1] - self._agent_location[1])
            )
            theta = math.atan2(robot_target_y, robot_target_x)
            waypoint_distances.append(radius)
            waypoint_directions.append(theta)
        return (
            np.array(waypoint_distances, dtype=np.float32),
            np.array(waypoint_directions, dtype=np.float32),
        )
    
    def _rotate_costmap(self):
        """
        Rotate and crop local costmap to align with the robot's orientation.

        Returns:
            np.ndarray: Rotated and cropped costmap.
        """
        local_costmap_rotated = None
        if self._observation_mode in ["costmap", "humap"]:
            angle_deg = -np.degrees(self._agent_orientation)
            costmap = np.fliplr(self._local_costmap)
            costmap_2d = np.ascontiguousarray(costmap[:, :, 0])
            costmap_2d = costmap_2d.astype(np.float32)
            (h, w) = costmap_2d.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            rotated_costmap = cv2.warpAffine(costmap_2d, M, (w, h),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT)
            rotated_costmap = rotated_costmap[..., np.newaxis]
            crop_h = int(h * 6.0/8.0) 
            crop_w = int(w * 6.0/8.0)
            start_y = (h - crop_h) // 2
            start_x = (w - crop_w) // 2
            cropped_costmap = rotated_costmap[start_y:start_y+crop_h,
                                              start_x:start_x+crop_w, :]
            reshaped_costmap = cv2.resize(
                cropped_costmap,
                (120, 120),
                interpolation=cv2.INTER_LINEAR
            )
            reshaped_costmap = np.rot90(reshaped_costmap, k=3)
            local_costmap_rotated = reshaped_costmap[..., None]
        return local_costmap_rotated
    
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
                / (self.NUM_WAYPOINTS)
            )
            .astype(np.float32)
        )
        # Normalize angles from [-pi, pi] to [-1,1]
        observation["waypoint_directions"] = (
            (observation["waypoint_directions"] / math.pi)
            .astype(np.float32)
        )

        if self._observation_mode == "costmap":
            observation["costmap"] = (
                (observation["costmap"] / 100).astype(np.float32)
            )
            self.publish_costmap_as_img(observation["costmap"])
        if self._observation_mode == "humap":
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
        """
        Convert normalized action values to robot control commands.

        Args:
            normalized_action (np.ndarray): Normalized action array.
        Returns:
            Twist: Robot control command.
        """
        action_linear = (
            (self.MAX_LINEAR_VELOCITY * (normalized_action[0] + 1))
        ) / 2
        action_angular = (
            (self.ANGULAR_VELOCITY * (normalized_action[1] + 1)) +
            (-self.ANGULAR_VELOCITY * (1 - normalized_action[1]))
        ) / 2
        twist_msg = Twist()
        twist_msg.linear.x = action_linear
        twist_msg.angular.z = action_angular
        return(twist_msg)

class CalcTwistActionServer(NavEnv):
    """Action server for calculating twist commands using DRL-based path
    following."""
    def __init__(self, observation_mode="humap"):
        """
        Initialize the CalcTwist action server for DRL-based path following.

        Args:
            observation_mode (str): Type of observation space
            ('costmap', 'humap').
        """
        super().__init__(observation_mode)
        self._action_server = ActionServer(
            self,
            CalcTwist,
            'calc_twist',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )
        self.get_logger().info('CalcTwist action server started')

    def goal_callback(self, goal_request):
        """
        Handle goal requests for the CalcTwist action.

        Args:
            goal_request: The goal request object.
        Returns:
            GoalResponse: Response to the goal request.
        """
        self.get_logger().info('Received CalcTwist goal request')

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Handle cancel requests for the CalcTwist action.

        Args:
            goal_handle: The goal handle object.
        Returns:
            CancelResponse: Response to the cancel request.
        """
        self.get_logger().info('Received cancel request')
        if goal_handle.is_active:
            goal_handle.canceled()
            self.get_logger().info('Goal canceled')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """
        Execute the CalcTwist action to calculate and publish twist commands.

        Args:
            goal_handle: The goal handle object.
        Returns:
            CalcTwist.Result: The result of the action execution.
        """
        start_time = time.perf_counter()
        self.get_logger().info('Executing CalcTwist goal...')
        path = goal_handle.request.path
        self.global_path = path
        self._target_location = (
            self.global_path.poses[-1].pose.position.x,
            self.global_path.poses[-1].pose.position.y
        )
        feedback_msg = CalcTwist.Feedback()
        result = CalcTwist.Result()

        if rclpy.ok():
            self.get_robot_location()

            # Compute distance to final goal
            last_pose = self.global_path.poses[-1].pose.position
            dx = last_pose.x - self._agent_location[0]
            dy = last_pose.y - self._agent_location[1]
            dist_to_goal = math.hypot(dx, dy)

            feedback_msg.distance_to_goal = float(dist_to_goal)

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return CalcTwist.Result()

            self.get_floating_waypoints(self.global_path)
            path = goal_handle.request.path
            self.pub_path.publish(self.global_path)
            state = self.get_obs()
            obs_time = time.perf_counter()
            self.publish_markers()
            self.publish_map_as_img(state)
            pub_time = time.perf_counter()
            action, _ = self.model.predict(state, deterministic=True)
            twist = self.denormalize_action(action)
            feedback_msg.speed = float(abs(twist.linear.x))
            action_time = time.perf_counter()
            result.command = twist
            self.get_logger().info(f'Publishing command: {twist}')
            goal_handle.publish_feedback(feedback_msg)
        else:
            self.get_logger().error('rclpy is not ok')
            goal_handle.abort()
            return result

        # Check if the loop is running at more than 20 Hz
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > 1 / 20:
            self.get_logger().info(
                f'obs time {obs_time - start_time:.6f} seconds'
            )
            self.get_logger().info(
                f'pub time {pub_time - obs_time:.6f} seconds'
            )
            self.get_logger().info(
                f'action time {action_time - pub_time:.6f} seconds'
            )
            self.get_logger().info(
                f'elapsed time {elapsed_time:.6f} seconds'
            )

        goal_handle.succeed()
        return result


def main():
    """
    Main entry point for the CalcTwist action server node.
    """
    rclpy.init()
    executor = MultiThreadedExecutor()
    server = CalcTwistActionServer(observation_mode="humap")
    executor.add_node(server)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
