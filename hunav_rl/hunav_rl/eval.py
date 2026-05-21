"""Evaluation module for robot navigation performance assessment."""


import math
import os
import pickle
import random
import time
from functools import partial
import numpy as np
from scipy.signal import convolve2d
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from ament_index_python.packages import get_package_share_directory
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from sensor_msgs.msg import LaserScan
from hunav_msgs.srv import SuccessTrigger
from people_msgs.msg import People


def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion.

    Args:
        roll: Rotation around x-axis in radians.
        pitch: Rotation around y-axis in radians.
        yaw: Rotation around z-axis in radians.

    Returns:
        Tuple of quaternion components (w, x, y, z).
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)



class Evaluation(Node):
    """ROS node for evaluating robot navigation in crowded environments."""

    def __init__(self):
        """Initialize the evaluation node with parameters and services."""
        super().__init__('create_paths')
        self.declare_parameter('start_id', 0)
        self.declare_parameter('num_evaluations', 100)
        self.declare_parameter('world_name', 'hospital')
        self.declare_parameter('num_people', 5)
        self.declare_parameter('planner', 'DWA')
        self.planner = self.get_parameter('planner').value

        # Parameters
        self.MINIMUM_DIST_FROM_TARGET = 1.0
        self.MINIMUM_DIST_FROM_OBSTACLES = 0.22

        # State variables
        self._global_costmap = OccupancyGrid()
        self._nav2_goal_rejected = False
        self._global_path = None
        self._receive_global_path = False
        self._scan_visual_msg = None
        self._scan_collision_msg = None
        self._collision = False
        self._hit_person = False
        self._reached_goal = False
        self._canceled_nav2 = False
        self._robot_position = [0, 0, 0, 0]
        self._target_position = [0, 0, 0, 0]

        # Publisher
        self.path_pub = self.create_publisher(Path, '/path', 10)

        # Service clients
        self.client_state = self.create_client(
            SetEntityState, "set_entity_state"
        )
        self.client_reset = self.create_client(
            Empty, "/rtabmap/reset"
        )
        self.client_unpause = self.create_client(
            Empty, '/unpause_physics'
        )
        self.client_pause = self.create_client(
            Empty, '/pause_physics'
        )
        self.client_reset_simulation = self.create_client(
            Empty, '/reset_simulation'
        )
        self.trigger_client = self.create_client(
            SuccessTrigger, '/hunav_trigger_recording'
        )

        # Subscribers
        self.people_sub = self.create_subscription(
            People, 'people', self.people_callback, 1
        )
        self.robot_pose_sub = self.create_subscription(
            Odometry, '/odom', self.robot_pose_callback, 1
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 1
        )
        self.scan_visual_sub = self.create_subscription(
            LaserScan, '/scan_visual', self.scan_visual_callback, 1
        )
        self.scan_collision_sub = self.create_subscription(
            LaserScan, '/scan_collision', self.scan_collision_callback, 1
        )
        self.global_costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.global_costmap_callback,
            1
        )

        # Action clients
        self.path_client = ActionClient(
            self, ComputePathToPose, 'compute_path_to_pose'
        )
        if not self.path_client.wait_for_server(timeout_sec=10.0):
            self.terminal_warning(
                "Action server ComputePathToPose not available!"
            )
            return
        self._action_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.terminal_warning("Action server not available!")
            return

    def robot_pose_callback(self, msg):
        """Callback for robot pose updates.

        Args:
            msg: Odometry message containing pose and velocity.
        """
        self._robot_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        # Check if the robot is close to the target
        distance = math.sqrt(
            (self._robot_position[0] - self._target_position[0])**2 +
            (self._robot_position[1] - self._target_position[1])**2
        )
        if distance < self.MINIMUM_DIST_FROM_TARGET:
            self._reached_goal = True
            self._finished = True
            self.terminal_output("Robot reached the target position.")

    def people_callback(self, msg: People):
        """Callback for human agent state updates.

        Args:
            msg: People message containing pedestrian states.
        """
        self._people = msg

    def get_people(self):
        """Get the latest pedestrian states from the /people topic.

        Returns:
            list: List of (x, y) tuples representing people positions.
        """
        self._people = None
        while self._people is None:
            rclpy.spin_once(self)
        people = []
        for person in self._people.people:
            position = person.position
            people.append((position.x, position.y))
        return people

    def scan_callback(self, msg):
        """Callback for laser scan updates.Checks the distance to obstacles.

        Args:
            msg: LaserScan message containing range data.
        """
        if np.any(np.array(msg.ranges) < self.MINIMUM_DIST_FROM_OBSTACLES):
            min_distance = min(msg.ranges)
            self.terminal_output(
                f"Distance to closest obstacle: {min_distance}"
            )
            self.terminal_output("Robot hit an obstacle.")
            self._collision = True
            self._finished = True

    def scan_visual_callback(self, msg):
        """Callback for visual laser scan updates.

        Args:
            msg: LaserScan message containing visual range data.
        """
        self._scan_visual_msg = msg

    def scan_collision_callback(self, msg):
        """Callback for collision laser scan updates.

        Args:
            msg: LaserScan message containing collision range data.
        """
        self._scan_collision_msg = msg

    def global_costmap_callback(self, msg: OccupancyGrid):
        """Callback for global costmap updates.

        Args:
            msg: OccupancyGrid message containing global costmap data.
        """
        self._global_costmap = msg

    def create_paths(
        self,
        path_file,
        min_target_distance: float = 7.0,
        min_ped_distance: float = 2.0,
        ped_radius: float = 0.35,
    ):
        """Create and save evaluation paths to file.

        Args:
            path_file: Path to the file where paths will be saved.
            min_target_distance: Minimum distance between start and target.
            min_ped_distance: Minimum distance between start and pedestrians.
            ped_radius: Radius of pedestrians.

        Returns:
            tuple: (robot_start_pos, target_pos, global_path)
            for the created path.
        """
        # Randomize start and target positions.
        is_spawning_possible = False
        while is_spawning_possible == False:
            is_spawning_possible = True
            robot_start_pos = self.randomize_robot_location()
            target_pos = self.randomize_target_location()

            # Check the distance between start and goal positions
            distance = math.sqrt(
                (robot_start_pos[0] - target_pos[0])**2 +
                (robot_start_pos[1] - target_pos[1])**2
            )
            if distance < min_target_distance:
                is_spawning_possible = False
                
            people = self.get_people()
            for person in people:   
                if math.sqrt(
                        (robot_start_pos[0] - person[0])**2
                        + (robot_start_pos[1] - person[1])**2
                ) < (
                        min_ped_distance
                        + self.MINIMUM_DIST_FROM_OBSTACLES
                        + ped_radius
                ):
                    is_spawning_possible = False

        start_pose = PoseStamped()
        start_pose.header.frame_id = 'map'
        start_pose.pose.position.x = robot_start_pos[0]
        start_pose.pose.position.y = robot_start_pos[1]
        start_pose.pose.orientation.z = robot_start_pos[2]
        start_pose.pose.orientation.w = robot_start_pos[3]

        end_pose = PoseStamped()
        end_pose.header.frame_id = 'map'
        end_pose.pose.position.x = target_pos[0]
        end_pose.pose.position.y = target_pos[1]
        quaternion = quaternion_from_euler(0, 0, 0)
        end_pose.pose.orientation.w = quaternion[0]
        end_pose.pose.orientation.x = quaternion[1]
        end_pose.pose.orientation.y = quaternion[2]
        end_pose.pose.orientation.z = quaternion[3]

        # Send goal to compute the path
        self.send_goal_cptp(start_pose, end_pose)
        while not self._receive_global_path:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.terminal_output("Waiting for global path...")

        # Reset flag for next use
        self._receive_global_path = False

        if len(self._global_path.poses) >= 10:
            target_waypoint = self._global_path.poses[9].pose.position
            dx = target_waypoint.x - robot_start_pos[0]
            dy = target_waypoint.y - robot_start_pos[1]
            yaw = math.atan2(dy, dx)
            quaternion = quaternion_from_euler(0, 0, yaw)
            robot_start_pos[2] = quaternion[3]  # z component of quaternion
            robot_start_pos[3] = quaternion[0]  # w component of quaternion

        data_entry = {
            "start": robot_start_pos,
            "goal": target_pos,
            "global_path": self._global_path
        }
        with open(path_file, 'ab') as f:
            pickle.dump(data_entry, f)
        self.terminal_output(
            f"Start and Goal positions and path saved to {path_file}"
        )
        return robot_start_pos, target_pos, self._global_path

    def load_paths(self, path, index):
        """Load evaluation paths from file.

        Args:
            path: Path to the file containing saved paths.
            index: Index of the path to load.

        Returns:
            tuple: (robot_start_pos, target_pos, global_path) if successful,
            None otherwise.
        """
        if not os.path.exists(path):
            self.terminal_output("Path file does not exist.")
            return None
        data_list = []
        with open(path, 'rb') as f:
            while True:
                try:
                    entry = pickle.load(f)
                    data_list.append(entry)
                except EOFError:
                    break
        if index < len(data_list):
            robot_start_pos = data_list[index]["start"]
            target_pos = data_list[index]["goal"]
            self._global_path = data_list[index]["global_path"]
            self._receive_global_path = True

            if len(self._global_path.poses) >= 10:
                target_waypoint = self._global_path.poses[9].pose.position
                dx = target_waypoint.x - robot_start_pos[0]
                dy = target_waypoint.y - robot_start_pos[1]
                yaw = math.atan2(dy, dx)
                quaternion = quaternion_from_euler(0, 0, yaw)
                robot_start_pos[2] = quaternion[3]  # z component of quaternion
                robot_start_pos[3] = quaternion[0]  # w component of quaternion


            self.terminal_output(f"Loaded path {index} from {path}")
            return robot_start_pos, target_pos, self._global_path
        else:
            self.terminal_output("Index out of range.")
            return None
    
    def cancel_done_callback(self, future):
        """Callback for handling the result of a goal cancel request.

        Args:
            future: The future object containing the cancel response.
        """
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.terminal_output("Goal successfully cancelled.")
        else:
            self.terminal_output(
                "Goal cancel request rejected or no goal "
                "to cancel."
            )
        self._canceled_nav2 = True

    def reset(self, robot_start_pos, target_position):
        """Resets the simulation and robot state for evaluation.

        Args:
            robot_start_pos (list): The starting pose of the robot.
            target_position (list): The target position for the robot.
        """
        self.done_pause = False
        self.done_reset_simulation = False
        self.done_robot_state = False
        self.done_reset_rtabmap = False
        # cancel the action if it is still active
        if hasattr(self, "_ntp_goal_handle"):
            cancel_future = self._ntp_goal_handle.cancel_goal_async()
            self.terminal_output("Cancelling active NavigateToPose goal.")
            cancel_future.add_done_callback(self.cancel_done_callback)
            while not self._canceled_nav2:
                rclpy.spin_once(self, timeout_sec=0.1)
            self.terminal_output("Canceld active NavigateToPose goal.")
            
        self.call_pause_physics()
        while not self.done_pause:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.call_reset_simulation()
        while not self.done_reset_simulation:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Robot start position: " + str(robot_start_pos))
        self.call_set_robot_state_service(robot_start_pos)
        self.call_set_target_state_service(target_position)
        while not self.done_robot_state:
            rclpy.spin_once(self, timeout_sec=0.1)
        while not self.done_reset_rtabmap:
            self.reset_rtabmap()
            rclpy.spin_once(self, timeout_sec=0.1)
            
        self._finished = False

    def start_stop_evaluation(
        self, start=True, reached_goal=False,
        hit_person=False, experiment_tag="0"
    ):
        """Starts or stops the evaluation recording.

        Args:
            start (bool): Whether to start or stop evaluation.
            reached_goal (bool): Whether the goal was reached.
            hit_person (bool): Whether a person was hit.
            experiment_tag (str): Tag for the experiment.
        """
        while not self.trigger_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Waiting for /hunav_trigger_recording service..."
            )
        request = SuccessTrigger.Request()

        if not start:
            self.get_logger().info("Start is False.")
        else:
            self.get_logger().info("Start is True.")

        if not start:
            request.reached_goal = reached_goal
            if self._finished and not self._collision:
                request.object_hit = 'None'
                self.get_logger().info("Object hit: None")
            elif hit_person:
                self.get_logger().info("Object hit: Person")
                request.object_hit = 'Person'
            else:
                request.object_hit = 'Object'
                self.get_logger().info("Object hit: Object")
        request.experiment_tag = str(experiment_tag)
        self.get_logger().info(f"Request reached_goal: {request.reached_goal}")
        self.get_logger().info(f"Request object_hit: {request.object_hit}")
        self.get_logger().info(
            f"Request experiment_tag: {request.experiment_tag}"
        )
        future = self.trigger_client.call_async(request)
        future.add_done_callback(self.response_callback_trigger)

    def response_callback_trigger(self, future):
        """Callback for handling the response from the trigger service.

        Args:
            future: The future object containing the service response.
        """
        try:
            response = future.result()
            if response.success:
                response_message = response.message
                self.get_logger().info(f"Service response: {response_message}")
                self._finished_start_stop = True
            else:
                self.get_logger().error("Failed to start/stop evaluation.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")
    


    def evaluate(self, start_pose, target_pose, index=0):
        """Runs the evaluation episode for a given start and target pose.

        Args:
            start_pose (list): The starting pose of the robot.
            target_pose (list): The target pose for navigation.
            index (int): The evaluation index/tag.

        Returns:
            bool: True if evaluation completed, False otherwise.
        """
        self.get_logger().info("Hello, world!")
        # unpause the physics simulation
        self.done_unpause = False
        self.call_unpause_physics()
        t = time.time()
        while not self.done_unpause:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - t > 10.0:
                self.get_logger().error(
                    "Timeout while waiting for unpause_physics response."
                )
                os._exit(1)
        # wait for observation (DRL_VO_reconstruction)
        self.get_logger().info(f"Planner: {self.planner}")
        if self.planner == "DRL-VO_Reconstruction":
            time.sleep(5)
        
        self._finished_start_stop = False
        self._reached_goal = False
        self._collision = False
        self._hit_person = False
        self.start_stop_evaluation(start=True, experiment_tag=str(index))
        t = time.time()
        while not self._finished_start_stop:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - t > 10.0:
                self.get_logger().error(
                    "Timeout while waiting for start/stop evaluation response."
                )
                os._exit(1)
        start_time = self.get_clock().now().nanoseconds / 1e9
        self.get_logger().info(
            f"Sending NavigateToPose action goal: {target_pose}"
        )
        self.navigate_to_pose(target_pose)
        self._collision = False
        self._finished = False

        self.get_logger().info("Starting evaluation...")
        while not self._finished:
            rclpy.spin_once(self)
            if self._nav2_goal_rejected:
                self._nav2_goal_rejected = False
                self.get_logger().warn(
                    "Goal was rejected. Sending NavigateToPose goal again."
                )
                self.navigate_to_pose(target_pose)
            if self._finished and (
                (self.get_clock().now().nanoseconds / 1e9) - start_time
            ) < 5.0:
                self.get_logger().warn(
                    "From start to finished less than 5 sec - reset finished"
                )
                self._finished = False
                self._collision = False
            if self._finished and (
                (self.get_clock().now().nanoseconds / 1e9) - start_time
            ) > 100.0:
                self.get_logger().warn(
                    "Episode took longer than 100 sec - skip."
                )
            # For stability restart if evaluation time exceeds 5 minutes
            if (
                (self.get_clock().now().nanoseconds / 1e9) - start_time
            ) > 300:
                self.get_logger().warn(
                    "Evaluation exceeded 5 minutes. Restarting."
                )
                self._finished = True
            self.call_unpause_physics()

        self.get_logger().info("Stopping evaluation...")

        self.get_logger().info("Canceling nav2 action goal")
        if hasattr(self, "_ntp_goal_handle"):
            cancel_future = self._ntp_goal_handle.cancel_goal_async()
            self.terminal_output("Cancelling active NavigateToPose goal.")
            cancel_future.add_done_callback(self.cancel_done_callback)
            while not self._canceled_nav2:
                rclpy.spin_once(self, timeout_sec=0.1)
            self.terminal_output("Canceld active NavigateToPose goal.")
        else:
            self.get_logger().warn("No active NavigateToPose goal to cancel.")

        # Check for people: if obstacle not in collision scan person was hit
        if not np.any(
            np.array(self._scan_collision_msg.ranges) <
            self.MINIMUM_DIST_FROM_OBSTACLES + 0.1
        ):
            self._hit_person = True

        if self._hit_person:
            self.get_logger().info("A person was hit by the robot.")
        else:
            self.get_logger().info("No person was hit by the robot.")

        self._finished_start_stop = False
        # success if the goal was reached without collision
        self.get_logger().info(f"Experiment tag: {str(index)}")
        self.start_stop_evaluation(
            start=False,
            reached_goal=(self._reached_goal and not self._collision),
            hit_person=self._hit_person,
            experiment_tag=str(index)
        )
        t = time.time()
        while not self._finished_start_stop:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - t > 10.0:
                self.get_logger().error(
                    "Timeout while waiting for start/stop evaluation "
                    "response."
                )
                os._exit(1)
        self.get_logger().info("Evaluation stopped.")
        self._finished = False
        return True

    def send_goal_cptp(self, start_pose, goal_pose):
        """Sends a ComputePathToPose goal to the path client.

        Args:
            start_pose (PoseStamped): The starting pose.
            goal_pose (PoseStamped): The goal pose.
        """
        # start and goal pose are PoseStamped
        goal_msg = ComputePathToPose.Goal()
        goal_msg.use_start = True
        goal_msg.start = start_pose
        goal_msg.goal = goal_pose
        self._send_goal_future = self.path_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(
            self.goal_response_callback_cptp
        )

    def goal_response_callback_cptp(self, future):
        """Callback for handling the response to the ComputePathToPose goal.
        
        Args:
            future: The future object containing the goal response.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.terminal_output(
                "Goal was rejected by the path action server."
            )
            self._nav2_goal_rejected = True
            return
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(
            self.get_result_callback_cptp
        )

    def get_result_callback_cptp(self, future):
        """Callback for handling the result of the ComputePathToPose action.
        
        Args:
            future: The future object containing the result.
        """
        result = future.result().result
        if result:
            self._global_path = result.path
            self._receive_global_path = True
            self.terminal_output("Global path received.")

    def navigate_to_pose(self, pose):
        """Sends a NavigateToPose goal to the action client.
        
        Args:
            pose (list): The target pose [x, y].
        """
        goal_msg = NavigateToPose.Goal()
        goal_pose_msg = PoseStamped()
        goal_pose_msg.header.frame_id = 'map'
        goal_pose_msg.pose.position.x = pose[0]
        goal_pose_msg.pose.position.y = pose[1]
        goal_msg.pose = goal_pose_msg
        self._send_goal_future_ntp = \
            self._action_client.send_goal_async(goal_msg)
        self._send_goal_future_ntp.add_done_callback(
            self.goal_response_callback_ntp
        )

    def goal_response_callback_ntp(self, future):
        """Callback for handling the response to the NavigateToPose goal.
        
        Args:
            future: The future object containing the goal response.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.terminal_output("Goal rejected by the path action server.")
            return
        self._ntp_goal_handle = goal_handle
        self._get_result_future_ntp = goal_handle.get_result_async()
        self._get_result_future_ntp.add_done_callback(
            self.get_result_callback_ntp
        )

    def get_result_callback_ntp(self, future):
        """Callback for handling the result of the NavigateToPose action.
        
        Args:
            future: The future object containing the result.
        """
        result = future.result().result
        if result:
            self.terminal_output("FINISHED: Action result received.")
            self._finished = True
            

    def randomize_target_location(self):
        """Randomizes the target location for evaluation.
        
        Returns:
            list: The randomized target location [x, y].
        """
        x, y = self.get_random_spawn_position()
        target_location = [x, y]
        return target_location

    def randomize_robot_location(self):
        """Randomizes the robot's starting location and orientation.
        
        Returns:
            list: The randomized robot location [x, y, z, w].
        """
        position_x, position_y = self.get_random_spawn_position()
        angle = float(math.radians(np.random.uniform(-180, 180)))
        quaternion = quaternion_from_euler(0, 0, angle)
        return [position_x, position_y, quaternion[3], quaternion[0]]

    def get_random_spawn_position(self):
        """Gets a random valid spawn position from the costmap.
        
        Returns:
            tuple: (x, y) coordinates of the spawn position.
        """
        while not self._global_costmap.data:
            self.terminal_output("Costmap is empty! Waiting...")
            rclpy.spin_once(self, timeout_sec=0.1)

        costmap = self._global_costmap
        width = costmap.info.width
        height = costmap.info.height
        resolution = costmap.info.resolution
        origin = costmap.info.origin  # geometry_msgs/Pose

        costmap_array = (
            np.array(costmap.data, dtype=np.uint8)
            .reshape((height, width))
        )
        free = (costmap_array == 0).astype(np.uint8)

        # Ensure not to spawn inside obstacles
        # by checking a 5-cell radius neighborhood
        padded = np.pad(free, pad_width=5, mode='constant', constant_values=0)
        radius = 5
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        kernel = (x**2 + y**2 <= radius**2).astype(np.uint8)
        neighborhood_sum = convolve2d(padded, kernel, mode='valid')
        valid_mask = (neighborhood_sum == kernel.sum())
        
        valid_y, valid_x = np.where(valid_mask)
        x_world = origin.position.x + (valid_x + 0.5) * resolution
        y_world = origin.position.y + (valid_y + 0.5) * resolution

        # Ensure spawn points are on one side of the crowd
        self.terminal_output("Using exclusion filter!")
        x_filter = (x_world <= -7.0) | (x_world >= 7.0)
        if not np.any(x_filter):
            self.get_logger().error(
                "No valid spawn points found with x outside -7 and 7!"
            )
            return (0.0, 0.0)
        x_final = x_world[x_filter]
        y_final = y_world[x_filter]
        idx = random.randint(0, len(x_final) - 1)
        return (x_final[idx], y_final[idx])

    def terminal_output(self, message):
        """Outputs an info message to the terminal.

        Args:
            message (str): The message to output.
        """
        self.get_logger().info(message)
    def terminal_warning(self, message):
        """Outputs a warning message to the terminal.

        Args:
            message (str): The warning message to output.
        """
        self.get_logger().warn(message)
    def terminal_error(self, message):
        """Outputs an error message to the terminal.

        Args:
            message (str): The error message to output.
        """
        self.get_logger().error(message)

    def call_set_robot_state_service(self, robot_pose=[-1, -3, -0.707, 0.707]):
        """Calls the set_entity_state service to set the robot's state.

        Args:
            robot_pose (list): The robot's pose [x, y, z, w].
        """
        while not self.client_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for set_entity_state service...")
        request = SetEntityState.Request()
        request.state.name = 'waffle_rl'
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
    
    def callback_set_robot_state(self, future):
        """Callback for handling the response from set_entity_state service
        for the robot. 

        Args:
            future: The future object containing the service response.
        """
        try:
            response = future.result()
            self.terminal_output("The Environment has been successfully reset")
            self.done_robot_state = True
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

    def call_set_target_state_service(self, target_pose=[-1, -3, 0, 0]):
        """Calls the set_entity_state service to set the target's state.
        
        Args:
            target_pose (list): The target's pose [x, y, z, w].
        """
        while not self.client_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for set_entity_state service...")
        request = SetEntityState.Request()
        request.state.name = 'target'
        request.state.pose.position.x = float(target_pose[0])
        request.state.pose.position.y = float(target_pose[1])
        request.state.pose.orientation.z = float(0)
        request.state.pose.orientation.w = float(0)
        request.state.twist.linear.x = float(0)
        request.state.twist.linear.y = float(0)
        request.state.twist.linear.z = float(0)
        request.state.twist.angular.x = float(0)
        request.state.twist.angular.y = float(0)
        request.state.twist.angular.z = float(0)
        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_target_state))
    
    def callback_set_target_state(self, future):
        """Callback for handling the response from set_entity_state service
        for the target.
        
        Args:
            future: The future object containing the service response.
        """
        try:
            response = future.result()
            self.terminal_output("The Environment has been successfully reset")
            self.done_target_state = True
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

    def call_unpause_physics(self):
        """Calls the /unpause_physics service to unpause the simulation."""
        if not self.client_unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Service /unpause_physics not available')
            return
        request = Empty.Request()
        self.future = self.client_unpause.call_async(request)
        self.future.add_done_callback(self.response_callback_unpause)
    
    def response_callback_unpause(self, future):
        """Callback for handling the response from /unpause_physics service.
        
        Args:
            future: The future object containing the service response.
        """
        try:
            future.result()
            self.done_unpause = True
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def call_pause_physics(self):
        """Calls the /pause_physics service to pause the simulation."""
        if not self.client_pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Service /pause_physics not available')
            return
        request = Empty.Request()
        self.future = self.client_pause.call_async(request)
        self.future.add_done_callback(self.response_callback_pause)

    def response_callback_pause(self, future):
        """Callback for handling the response from /pause_physics service.
        
        Args:
            future: The future object containing the service response.
        """
        try:
            future.result()
            self.get_logger().info(
                'Successfully called /pause_physics service'
            )
            self.done_pause = True
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def call_reset_simulation(self):
        """Calls the /reset_simulation service to reset the simulation."""
        if not self.client_reset_simulation.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /reset_simulation not available')
            return
        request = Empty.Request()
        self.future = self.client_reset_simulation.call_async(request)
        self.future.add_done_callback(self.response_callback_reset_simulation)

    def response_callback_reset_simulation(self, future):
        """Callback for handling the response from /reset_simulation service.
        
        Args:
            future: The future object containing the service response.
        """
        try:
            future.result()
            self.get_logger().info(
                'Successfully called /reset_simulation service'
            )
            self.done_reset_simulation = True
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def reset_rtabmap(self):
        """Calls the RTAB-Map reset service."""
        if not self.client_reset.service_is_ready():
            self.terminal_warning("Service not available, skipping reset.")
            self.call_unpause_physics()
            return

        request = Empty.Request()
        future = self.client_reset.call_async(request)
        future.add_done_callback(partial(self.callback_reset_rtabmap))
    
    def callback_reset_rtabmap(self, future):
        """Callback for handling the response from RTAB-Map reset service.
        
        Args:
            future: The future object containing the service response.
        """
        try:
            response = future.result()
            self.terminal_output("RTAB-Map has been successfully reset")
            self.done_reset_rtabmap = True
        except Exception as e:
            self.terminal_error("Service call failed: %r" % (e,))

def main(args=None):
    """Main function for running robot navigation evaluation.

    Initializes the ROS node, loads or creates evaluation paths, and runs
    navigation evaluations for the specified number of episodes. Handles
    path creation, robot reset, and evaluation execution.

    Args:
        args: Command line arguments passed to rclpy.init().
    """
    rclpy.init(args=args)
    node = Evaluation()
    start_id = node.get_parameter('start_id').value
    num_evaluations = node.get_parameter('num_evaluations').value
    world_name = node.get_parameter('world_name').value
    num_people = node.get_parameter('num_people').value
    planner = node.get_parameter('planner').value
    create_paths = False
    just_create_paths = False

    # Loading evaluation paths
    pkg_share_dir = get_package_share_directory('hunav_rl')
    ws_dir = os.path.abspath(
        os.path.join(pkg_share_dir, '..', '..', '..', '..')
    )
    evaluation_dir = os.path.abspath(
        os.path.join(ws_dir, 'src', 'drl-sfm', 'hunav_rl', 'evaluation')
    )
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    path_file = os.path.join(
        evaluation_dir, 
        f'{world_name}_{num_people}_{planner}',
        f'eval_paths_{world_name}.pkl'
    )
    if not os.path.exists(path_file):
        create_paths = True
        just_create_paths = True
        node.get_logger().info(
            f"eval_paths_{world_name}.pkl does not exist. Creating a new file."
        )
        with open(path_file, 'w') as f:
            pass
    else:
        node.get_logger().info(f"Loading eval_paths_{world_name}.pkl.")

    evaluation_folder = os.path.join(
        evaluation_dir, f"{world_name}_{num_people}_{planner}"
    )

    try:
        for i in range(start_id, start_id+num_evaluations):
            # Check whether the metrics file already exists
            metrics_steps_file = os.path.join(
                evaluation_folder, f"metrics_steps_{i}.txt"
            )
            node.get_logger().info(
                "Checking if metrics_steps file exists: "
                f"{metrics_steps_file}"
            )
            if os.path.exists(metrics_steps_file):
                node.get_logger().info(
                    f"Metrics file for path {i} already exists. Skipping."
                )
                continue

            node.get_logger().info(f"Evaluating path {i}...")
            is_path = False
            while not is_path:
                if create_paths:
                    # Create new paths
                    node.get_logger().info(
                        f"Creating new path for world {world_name}."
                    )
                    (robot_start_pos, target_pos, global_path) = \
                        node.create_paths(path_file)
                else:
                    # Load paths from the file
                    node.get_logger().info(
                        f"Loading path {i} from {path_file}."
                    )
                    robot_start_pos, target_pos, global_path = node.load_paths(
                        path_file, i)
                if len(global_path.poses) > 0:
                    is_path = True
                    skip = False
                else:
                    node.get_logger().warn("No valid path found.")
                    skip = True
                    time.sleep(1)
            if skip:   
                continue

            if not just_create_paths:
                node.reset(robot_start_pos, target_pos)
                node._target_position = target_pos
                first_pose_path = [
                    global_path.poses[0].pose.position.x,
                    global_path.poses[0].pose.position.y
                ]
                node.get_logger().info(
                    f"First pose of the path: {first_pose_path}"
                )
                node.call_unpause_physics()
                if not node.evaluate(robot_start_pos, target_pos, index=i):
                    node.get_logger().error(
                        "Evaluation failed (5 min timer). Restarting..."
                    )
                    break
            else:
                node.get_logger().info("Creating paths, skipping evaluation.")
                node.reset(robot_start_pos, target_pos)
                node.call_unpause_physics()

    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt. Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()