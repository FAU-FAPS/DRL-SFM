"""Robot controller base class and utilities for hunav_rl."""


from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from functools import partial
import numpy as np
import math
from gazebo_msgs.srv import SetEntityState
import os
from ament_index_python.packages import get_package_share_directory
import rclpy
from std_msgs.msg import Float32MultiArray
from rclpy.action import ActionClient
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
import random
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from people_msgs.msg import People
np.float = float
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped
from hunav_rl import lightsfm
import json

class RobotController(Node):
    """ROS node for controlling robot navigation with RL."""

    def __init__(self, env_id):
        """Initialize the robot controller.

        Args:
            env_id: Environment identifier for multi-environment training.
        """
        super().__init__('robot_controller')
        self.get_logger().info(
            "The robot controller node has just been created"
        )
        self.ENV_ID = env_id
        self.ROBOT_NAME = "waffle_rl"
        # Action publisher
        self.action_pub = self.create_publisher(
            Twist, 'cmd_vel', 10
        )
        self.reward_pub = self.create_publisher(
            Float32MultiArray, 'reward', 10
        )
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'initialpose', 10
        )
        self.marker_pub = self.create_publisher(
            Marker, 'observation', 10
        )
        self.image_pub_costmap = self.create_publisher(
            Image, 'costmap_image', 10
        )
        self.pose_sub = self.create_subscription(
            Odometry, 'odom', self.pose_callback, 1
        )
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 1
        )
        self.agents_sub = self.create_subscription(
            People, 'people', self.human_states_callback, 1
        )
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/costmap/costmap',
            self.local_costmap_callback,
            1
        )
        self.global_costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.global_costmap_callback,
            1
        )
        self.client_state = self.create_client(
            SetEntityState, "set_entity_state"
        )
        self.client_reset = self.create_client(
            Empty, "/rtabmap/reset"
        )
        self.path_client = ActionClient(
            self, ComputePathToPose, 'compute_path_to_pose'
        )
        if not self.path_client.wait_for_server(timeout_sec=10.0):
            return
        self.reset_laser_time = 0.0
        self._collision = False
        self._reached_target = False
        self._agent_location = np.array([np.float32(-1),np.float32(-3)])
        self._laser_reads = np.array([np.float32(10)] * 60)
        self._local_costmap = None
        self._local_costmap_info = None
        self._agents = People()
        self._global_costmap = OccupancyGrid()
        self._global_path = Path()
        self._received_global_path = False
        self.map_obstacles = []
        self.initialize_map_obstacles()

    def load_config(self):
        """Load config from training_scenario.json and set flags."""
        pkg_share_dir = get_package_share_directory('hunav_rl')
        ws_dir = os.path.abspath(
            os.path.join(pkg_share_dir, '..', '..', '..', '..')
        )
        pkg_dir = os.path.join(ws_dir, 'src', 'hunav_rl')
        config_file = os.path.join(pkg_dir, 'config', 'training_scenario.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            if "scenario" in config:
                scenario = config["scenario"]
        else:
            self.terminal_warning(
                f"Config file {config_file} does not exist. "
                "Using default config."
            )
            scenario = []
        if "sfm_prediction" in scenario:
            self._use_sfm_prediction = True
        else:
            self._use_sfm_prediction = False
        if "cost_as_reward" in scenario:
            self._use_cost_as_reward = True
        else:
            self._use_cost_as_reward = False
        if "velocity_obstacles" in scenario:
            self._use_vo_reward = True
        else:
            self._use_vo_reward = False
        if "angle_reward" in scenario:
            self._use_angle_reward = True
        else:
            self._use_angle_reward = False

    def send_velocity_command(self, velocity):
        """Send velocity command to the robot.

        Args:
            velocity: Array-like [linear_velocity, angular_velocity].
        """
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.angular.z = float(velocity[1])
        self.action_pub.publish(msg)

    def send_reward(self, reward):
        """Publish the reward to the reward topic.

        Args:
            reward: List of reward values to publish.
        """
        msg = Float32MultiArray()
        msg.data = [float(value) for value in reward]
        self.reward_pub.publish(msg)

    def pose_callback(self, msg: Odometry):
        """Callback for robot pose updates.

        Args:
            msg: Odometry message containing pose and velocity.
        """
        self._agent_location = np.array([
            np.float32(msg.pose.pose.position.x),
            np.float32(msg.pose.pose.position.y)
        ])
        self._agent_orientation = 2 * math.atan2(
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        self._agent_lin_vel = np.array([
            np.float32(msg.twist.twist.linear.x),
            np.float32(msg.twist.twist.linear.y)
        ])
        self._agent_ang_vel = np.float32(msg.twist.twist.angular.z)
        if hasattr(self, 'MINIMUM_DIST_FROM_TARGET') and \
           hasattr(self, '_target_location'):
            distance = math.dist(
                self._agent_location[:2],
                self._target_location[:2]
            )
            if distance < self.MINIMUM_DIST_FROM_TARGET:
                self._reached_target = True
        self._done_pose = True

    def laser_callback(self, msg: LaserScan):
        """Callback for laser scan updates.

        Args:
            msg: LaserScan message containing range data.
        """
        scan_time = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
        if scan_time >= self.reset_laser_time:
            laser_ranges = np.array(msg.ranges)
            if hasattr(self, 'MINIMUM_DIST_FROM_OBSTACLES'):
                if (any(laser_ranges < self.MINIMUM_DIST_FROM_OBSTACLES)):
                    self._collision = True
            self._laser_reads = np.min(laser_ranges.reshape(-1, 6), axis=1)
            self._laser_reads[self._laser_reads == np.inf] = np.float32(10)
            self._done_laser = True
        else:
            self.terminal_output("Laser reset time: " +
                                  str(self.reset_laser_time))
            self.terminal_output("Laser scan time: " + str(scan_time))
    
    def update_laser_reset_time(self):
        """Update the laser reset time to the current simulation time."""
        current_time = self.get_clock().now()
        self.reset_laser_time = current_time.nanoseconds 

    def local_costmap_callback(self, msg: OccupancyGrid):
        """Callback for local costmap updates.

        Args:
            msg: OccupancyGrid message containing costmap data.
        """
        empty = False
        empty = not any(np.array(msg.data) > 0.2)
        if not empty:
            self._local_costmap = np.array(msg.data).reshape(
                (msg.info.height, msg.info.width, 1)
            )
            self._local_costmap_info = msg.info
            self._received_new_data = True
        else:
            self.terminal_output(
                "Local costmap is empty, skipping update."
            )

    def global_path_callback(self, msg: Path):
        """Callback for global path updates.

        Args:
            msg: Path message containing global path.
        """
        if len(msg.poses) != 0:
            self._global_path = msg
            self._received_global_path = True

    def call_set_robot_state_service(self, robot_pose=[-1, -3, -0.707, 0.707]):
        """Call service to set robot state in simulation.

        Args:
            robot_pose: List [x, y, z, w] for robot position and orientation.
        """
        # check if position is too close to human agent
        self._human_too_close = False
        closest_distance = float('inf')
        human_states = self.get_human_states()
        print_flag = True
        for agent in human_states.people:
            agent_position = (agent.position.x, agent.position.y)
            distance = np.linalg.norm(
                np.array([robot_pose[0], robot_pose[1]])
                - np.array([agent_position[0], agent_position[1]])
            )
            if distance < closest_distance:
                closest_distance = distance
        if closest_distance < 1.5:
            if print_flag:
                print_flag = False
            self._reset_possible = False
            self._human_too_close = True
            return
        max_iterations = 100
        iteration = 0
        while not self.client_state.wait_for_service(1.0):
            if iteration >= max_iterations:
                self._reset_possible = False
                return
            self.terminal_warning("Waiting for service...")
            iteration += 1
        request = SetEntityState.Request()
        request.state.name = self.ROBOT_NAME
        request.state.pose.position.x = float(robot_pose[0])
        request.state.pose.position.y = float(robot_pose[1])
        request.state.pose.orientation.z = float(robot_pose[2])
        request.state.pose.orientation.w = float(robot_pose[3])
        request.state.twist.linear.x = float(0)
        request.state.twist.linear.y = float(0)
        request.state.twist.linear.z = float(0)
        request.state.twist.angular.x = float(0)
        request.state.twist.angular.y = float(0)
        request.state.twist.angular.z = float(0)
        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_robot_state))
        self._reset_possible = True

    def publish_initial_pose(self, robot_pose):
        """Publish the initial pose of the robot.

        Args:
            robot_pose: List [x, y, z, w] for robot position and orientation.
        """
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.pose.pose.position.x = float(robot_pose[0])
        initial_pose.pose.pose.position.y = float(robot_pose[1])
        initial_pose.pose.pose.position.z = 0.0
        initial_pose.pose.pose.orientation.x = 0.0
        initial_pose.pose.pose.orientation.y = 0.0
        initial_pose.pose.pose.orientation.z = float(robot_pose[2])
        initial_pose.pose.pose.orientation.w = float(robot_pose[3])
        self.initialpose_pub.publish(initial_pose)
    
    def reset_rtabmap(self):
        """Reset the RTAB-Map SLAM system via service call."""
        if not self.client_reset.service_is_ready():
            self.terminal_warning("Service not available, skipping reset.")
            self._done_reset_rtabmap = True
            return

        request = Empty.Request()
        future = self.client_reset.call_async(request)
        future.add_done_callback(partial(self.callback_reset_rtabmap))
    
    def callback_reset_rtabmap(self, future):
        """Callback for RTAB-Map reset service response.

        Args:
            future: Future object for async service call.
        """
        try:
            response = future.result()
            self.terminal_output("RTAB-Map has been successfully reset")
        except Exception as e:
            self.terminal_error("Service call failed: %r" % (e,))
        self._done_reset_rtabmap = True

    def human_states_callback(self, msg: People):
        """Callback for human agent state updates.

        Args:
            msg: People message containing agent states.
        """
        self._agents = msg

    def is_people_topic_active(self):
        """Check if the /people topic has active publishers.

        Returns:
            bool: True if topic is active, False otherwise.
        """
        publishers_info = self.get_publishers_info_by_topic('/people')
        return len(publishers_info) > 0
    
    def get_human_states(self):
        """Get the latest human agent states from the /people topic.

        Returns:
            People: Latest agent states.
        """
        if not self.is_people_topic_active():
            return People()
        self._agents = None
        while self._agents is None:
            self.terminal_warning("waiting for people message")
            self.send_velocity_command([0.0, 0.0])
            rclpy.spin_once(self)
        return self._agents

    def global_costmap_callback(self, msg: OccupancyGrid):
        """Callback for global costmap updates.

        Args:
            msg: OccupancyGrid message containing global costmap data.
        """
        self._global_costmap = msg
        if hasattr(self, 'global_costmap_sub'):
            self.destroy_subscription(self.global_costmap_sub)
            self.get_logger().info(
                "Unsubscribed global_costmap after first message."
            )

    def callback_set_robot_state(self, future):
        """Callback for robot state set service response.

        Args:
            future: Future object for async service call.
        """
        try:
            response= future.result()
            self.terminal_output("The Environment has been successfully reset")
            self._done_set_rob_state = True
        except Exception as e:
            self.terminal_error("Service call failed: %r" % (e,))

    def callback_set_target_state(self, future):
        """Callback for target state set service response.

        Args:
            future: Future object for async service call.
        """
        try:
            response= future.result()
        except Exception as e:
            self.terminal_error("Service call failed: %r" % (e,))
        self._done_set_tar_state = True
    
    def send_goal_cptp(self, start_pose, goal_pose):
        """Send a goal to the ComputePathToPose action server.

        Args:
            start_pose: PoseStamped for start position.
            goal_pose: PoseStamped for goal position.
        """
        goal_msg = ComputePathToPose.Goal()
        goal_msg.use_start = True
        goal_msg.start = start_pose
        goal_msg.goal = goal_pose
        self._send_goal_future = self.path_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(
            self.goal_respone_callback_cptp
        )

    def goal_respone_callback_cptp(self, future):
        """Callback for ComputePathToPose goal response.

        Args:
            future: Future object for async action call.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            return
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(
            self.get_result_callback_cptp
        )

    def get_result_callback_cptp(self, future):
        """Callback for ComputePathToPose result response.

        Args:
            future: Future object for async action call.
        """
        self._global_path = future.result().result.path
        self._received_global_path = True
        self.terminal_output("Global path received")
    
    def get_path(self, robot_start_pos, target_pos, old_path=None):
        """Request a global path from start to target position.

        Args:
            robot_start_pos: List [x, y, theta] for start position.
            target_pos: List [x, y] for target position.
            old_path: Optional Path to use if new path is invalid.
        """
        self._received_global_path = False
        start_pose = PoseStamped()
        start_pose.header.frame_id = 'map'
        start_pose.pose.position.x = float(robot_start_pos[0])
        start_pose.pose.position.y = float(robot_start_pos[1])
        quaternion = quaternion_from_euler(0, 0, float(robot_start_pos[2]))
        start_pose.pose.orientation.w = quaternion[0]
        start_pose.pose.orientation.x = quaternion[1]
        start_pose.pose.orientation.y = quaternion[2]
        start_pose.pose.orientation.z = quaternion[3]
        end_pose = PoseStamped()
        end_pose.header.frame_id = 'map'
        end_pose.pose.position.x = float(target_pos[0])
        end_pose.pose.position.y = float(target_pos[1])
        quaternion = quaternion_from_euler(0, 0, 0)
        end_pose.pose.orientation.w = quaternion[0]
        end_pose.pose.orientation.x = quaternion[1]
        end_pose.pose.orientation.y = quaternion[2]
        end_pose.pose.orientation.z = quaternion[3]
        self.send_goal_cptp(start_pose, end_pose)
        print_flag = True
        while not self._received_global_path:
            rclpy.spin_once(self)
            if print_flag:
                self.terminal_output("Waiting for global path")
                print_flag = False
        self._received_global_path = False
        if self._global_path is None or len(self._global_path.poses) == 0:
            if old_path is None or len(old_path.poses) == 0:
                self.get_logger().error("No valid path received, again")
            else:
                self.terminal_warning("No valid path received, using old path")
                self._global_path = old_path

    def is_long_enough(self, path, distance):
        """Check if the path is long enough for the required distance.

        Args:
            path: Path object to check.
            distance: Required path length.

        Returns:
            bool: True if path is long enough, False otherwise.
        """
        if len(path.poses) == 0:
            self._global_path = Path()
            return False
        positions = np.array([
            [p.pose.position.x, p.pose.position.y] for p in path.poses
        ])
        segs = positions[1:] - positions[:-1]
        seg_lengths = np.linalg.norm(segs, axis=1)
        cum_dists = np.cumsum(seg_lengths)
        idx = np.searchsorted(cum_dists, distance)
        if idx < len(cum_dists):
            path.poses = path.poses[:idx+1]
            self._global_path = path
            return True
        else:
            self._global_path = Path()
            return False

    def get_floating_waypoints(self, path, lookahead_distance=1.0):
        """Calculate floating waypoints ahead of the robot on the path.

        Args:
            path: Path object to extract waypoints from.
            lookahead_distance: Distance ahead for each waypoint.

        Returns:
            list: List of (x, y) waypoint tuples.
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
            # Find index of closest point on path to robot.
            dists = np.linalg.norm(poses - robot_pos, axis=1)
            closest_idx = np.argmin(dists)
            # Sub-path from the closest point to the end.
            sub_poses = poses[closest_idx:]
            if len(sub_poses) < 2:
                goal = sub_poses[-1]
            else:
                segs = sub_poses[1:] - sub_poses[:-1]
                seg_dists = np.linalg.norm(segs, axis=1)
                # Cumulative distances along the sub-path.
                cum_dists = np.concatenate(([0], np.cumsum(seg_dists)))
                if cum_dists[-1] < lookahead_distance:
                    # If path is shorter than lookahead, select the last pose.
                    goal = sub_poses[-1]
                else:
                    # Find idx where cumulative distance > lookahead_distance.
                    idx = np.searchsorted(cum_dists, lookahead_distance)
                    prev_dist = cum_dists[idx - 1]
                    seg_length = cum_dists[idx] - prev_dist
                    ratio = (
                        (lookahead_distance - prev_dist) / seg_length
                        if seg_length else 0
                    )
                    goal = (sub_poses[idx - 1] +
                            ratio * (sub_poses[idx] - sub_poses[idx - 1]))
            waypoints.append((goal[0], goal[1]))
            robot_pos = goal
        return waypoints

    def get_random_spawn_position(self):
        """Get a random free position from the global costmap.

        Returns:
            tuple: (x, y) coordinates in world frame.
        """
        costmap = self._global_costmap
        width = costmap.info.width
        resolution = costmap.info.resolution
        origin = costmap.info.origin
        free_indices = np.where(np.array(costmap.data) == 0)[0]
        if free_indices.size == 0:
            self.get_logger().error("No free cells available for spawn!")
            return
        selected_idx = random.choice(free_indices)
        x_cell = selected_idx % width
        y_cell = selected_idx // width
        x_world = origin.position.x + (x_cell + 0.5) * resolution
        y_world = origin.position.y + (y_cell + 0.5) * resolution

        return (x_world, y_world)

    def initialize_map_obstacles(self):
        """Initialize map obstacles from the global costmap."""
        while not self._global_costmap.data:
            rclpy.spin_once(self)
        resolution = self._global_costmap.info.resolution
        origin_x = self._global_costmap.info.origin.position.x
        origin_y = self._global_costmap.info.origin.position.y
        width = self._global_costmap.info.width
        height = self._global_costmap.info.height
        data = np.array(
            self._global_costmap.data, dtype=np.int32
        ).reshape((height, width))
        mask = data >= 50
        rows, cols = np.nonzero(mask)
        x_vals = origin_x + (cols + 0.5) * resolution
        y_vals = origin_y + (rows + 0.5) * resolution
        self.map_obstacles.extend([
            lightsfm.Vector2d(x, y) for x, y in zip(x_vals, y_vals)
        ])
        self.get_logger().info(
            "SFM Map obstacles initialized: {} obstacles found.".format(
            len(self.map_obstacles)))
    
    def compute_sfm_force(self, dt=None):
        """Compute the social force model forces for the robot and agents.

        Args:
            dt: Optional time delta for prediction.

        Returns:
            tuple: (global_force, obstacle_force, social_force, prediction)
        """
        prediction = None
        if dt is not None:
            if dt <= 0.0:
                self.old_sfm_instance = None
                self.old_sfm_robot = None
            if (self.old_sfm_instance is not None and
                    self.old_sfm_robot is not None):
                self.old_sfm_instance.update_position_for_agent(
                    self.old_sfm_robot, dt)
                prediction = np.array([
                    self.old_sfm_robot.position.getX(),
                    self.old_sfm_robot.position.getY()
                ])

        self.old_sfm_instance = None
        self.old_sfm_robot = None
        # create robot agent
        rob_pos = self._agent_location[0:2]
        robot_pos = lightsfm.Vector2d(rob_pos[0], rob_pos[1])
        robot_yaw = lightsfm.Angle.fromRadian(self._agent_orientation)
        robot_lin_vel = math.sqrt(
            self._agent_lin_vel[0] ** 2 + self._agent_lin_vel[1] ** 2
        )
        robot_vel_lin = float(robot_lin_vel)
        robot_vel_ang = float(self._agent_ang_vel)
        agent_robot = lightsfm.Agent(
            robot_pos, robot_yaw, robot_vel_lin, robot_vel_ang
        )
        agent_robot.teleoperated = False
        agent_robot.desiredVelocity = self.MAX_LINEAR_VELOCITY
        agent_robot.radius = self.MINIMUM_DIST_FROM_OBSTACLES
        # create goal
        robot_goal = lightsfm.Goal()
        if dt is not None:
            robot_goal.center = lightsfm.Vector2d(
                self.waypoints[0][0], self.waypoints[0][1]
            )
        else:
            robot_goal.center = lightsfm.Vector2d(
                self._target_location[0],
                self._target_location[1]
            )
        robot_goal.radius = self.MINIMUM_DIST_FROM_TARGET
        agent_robot.goals = [robot_goal]
        agent_robot.obstacles1.clear()
        # Extract obstacles from local costmap
        local_costmap = self._local_costmap
        if local_costmap.size > 0 and hasattr(self, '_local_costmap_info'):
            resolution = self._local_costmap_info.resolution
            origin_x = self._local_costmap_info.origin.position.x
            origin_y = self._local_costmap_info.origin.position.y
            if local_costmap.ndim == 3:
                obstacle_data = local_costmap[:, :, 0]
            else:
                obstacle_data = local_costmap
            obstacle_mask = obstacle_data > 99
            rows, cols = np.nonzero(obstacle_mask)
            if len(rows) > 0:
                # Convert cell coordinates to world coordinates
                x_world = origin_x + (cols + 0.5) * resolution
                y_world = origin_y + (rows + 0.5) * resolution
                # Filter obstacles within 3.0m of robot
                robot_pos_ = np.array([
                    agent_robot.position.getX(),
                    agent_robot.position.getY()
                ])
                obstacle_coords = np.column_stack((x_world, y_world))
                dists = np.linalg.norm(obstacle_coords - robot_pos_, axis=1)
                close_indices = np.where(dists <= 3.0)[0]
                # Add close obstacles to agent
                for idx in close_indices:
                    obstacle_point = lightsfm.Vector2d(
                        x_world[idx], y_world[idx]
                    )
                    agent_robot.obstacles1.append(obstacle_point)
        
        human_agents = []
        for agent in self._agents.people:
            dx = agent.position.x - agent_robot.position.getX()
            dy = agent.position.y - agent_robot.position.getY()
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= 3.0:
                agent_pos = lightsfm.Vector2d(
                    agent.position.x, agent.position.y
                )
                agent_yaw = lightsfm.Angle.fromRadian(
                    math.atan2(agent.velocity.y, agent.velocity.x)
                )
                agent_vel_ang = float(0.0)
                agent_vel_lin = float(
                    math.sqrt(agent.velocity.x ** 2 + agent.velocity.y ** 2)
                )
                agent_human = lightsfm.Agent(
                    agent_pos, agent_yaw, agent_vel_lin, agent_vel_ang
                )
                ego_goal = lightsfm.Goal()
                division_factor = max(
                    abs(agent.velocity.x), abs(agent.velocity.y)
                )
                if division_factor != 0:
                    goal_x = (agent.position.x +
                              agent.velocity.x / division_factor * 10.0)
                    goal_y = (agent.position.y +
                              agent.velocity.y / division_factor * 10.0)
                else:
                    goal_x = agent.position.x
                    goal_y = agent.position.y
                ego_goal.center = lightsfm.Vector2d(goal_x, goal_y)
                ego_goal.radius = 0.6
                agent_human.goals.clear()
                agent_human.goals.append(ego_goal)
                human_agents.append(agent_human)
        agents = [agent_robot] + human_agents
        sfm_instance = lightsfm.SocialForceModel.get_instance()
        sfm_instance.compute_forces_for_agent(agent_robot, agents)
        self.old_sfm_instance = sfm_instance
        self.old_sfm_robot = agent_robot
        return prediction
    
    def publish_target_location(self):
        """Publish the target location as a marker in RViz."""
        goal_pose = self._target_location
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = goal_pose[0]
        marker.pose.position.y = goal_pose[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)

    def publish_waypoints(self):
        """Publish the agent's waypoints as markers in RViz."""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "agent_waypoints"
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
            r_val = r_val * self.NUM_WAYPOINTS * self.LOOKAHEAD_DISTANCE
            theta = angle[i]
            theta = theta * math.pi
            x = r_val * math.cos(theta)
            y = r_val * math.sin(theta)
            pt = Point(x=x, y=y, z=0.0)
            points.append(pt)
        marker.points = points
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        self.marker_pub.publish(marker)
    
    def publish_robot_location(self):
        """Publish the robot's location as a marker in RViz."""
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

    def publish_sfm_prediction(self, sfm_prediction):
        """Publish the SFM prediction as a marker in RViz.

        Args:
            sfm_prediction: Array-like [x, y] prediction coordinates.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "sfm_prediction"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(sfm_prediction[0])
        marker.pose.position.y = float(sfm_prediction[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.45
        marker.scale.y = 0.45
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        self.marker_pub.publish(marker)

    def publish_humans_location(self):
        """Publish the locations of human agents as markers in RViz."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "humans"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        points = []
        for agent in self._agents.people:
            pt = Point(x=agent.position.x, y=agent.position.y, z=0.0)
            points.append(pt)
        marker.points = points
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)

    def publish_markers(self):
        """Publish all relevant markers (target, robot, obs, humans)
        in RViz."""
        self.publish_target_location()
        self.publish_robot_location()
        self.publish_waypoints()
        self.publish_humans_location()
        
    def publish_costmap_as_img(self, costmap):
        """Publish the costmap as an RGB image for visualization.

        Args:
            costmap: Numpy array representing the costmap.
        """
        mask1 = (costmap[..., 1:] >= 0.49) & (costmap[..., 1:] <= 0.51)
        costmap[..., 1:][mask1] = 0.0
        
        mask = (costmap[..., 0] >= 0.01) & mask1[..., 0]
        costmap[..., 1][mask] = costmap[..., 0][mask]
        costmap[..., 2][mask] = costmap[..., 0][mask]

        if costmap.ndim == 2 or (costmap.ndim == 3 and costmap.shape[-1] == 1):
            costmap_rgb = np.stack((costmap, costmap, costmap), axis=-1)
        else:
            costmap_rgb = costmap
        costmap_rgb = ((1.0 - costmap_rgb) * 255).astype(np.uint8)
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

    def terminal_output(self, info_str):
        """Log info messages to the ROS logger.

        Args:
            info_str: String to log as info.
        """
        if self.ENV_ID == 0:
            self.get_logger().info(info_str)
    
    def terminal_warning(self, info_str):
        """Log warning messages to the ROS logger.

        Args:
            info_str: String to log as warning.
        """
        self.get_logger().warn(info_str)
    
    def terminal_error(self, info_str):
        """Log error messages to the ROS logger.

        Args:
            info_str: String to log as error.
        """
        self.get_logger().error(info_str)