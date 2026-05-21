"""Callbacks for RL training in hunav_rl."""


import os
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from stable_baselines3.common.callbacks import BaseCallback

class StartStopPublisherThread(threading.Thread):
    """
    Thread to set ROS_DOMAIN_ID, initialize rclpy, create node with a publisher
    on /prevent_pause to pause and unpause gazebo, and spin.
    """
    def __init__(self, env_index: int, first_ros_domain_id: int):
        """Initialize the publisher thread.
        
        Args:
            env_index: Index of the environment for this thread.
            first_ros_domain_id: Base ROS domain ID to calculate environment-
            specific domain.
        """
        super().__init__(daemon=True)
        self.env_index = env_index
        self.first_ros_domain_id = first_ros_domain_id
        self.node = None
        self.publisher = None
        self._ready = threading.Event()

    def run(self):
        """Run the thread to create ROS node and publisher."""
        os.environ["ROS_DOMAIN_ID"] = str(
            self.first_ros_domain_id + self.env_index
        )
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node(f"startstop_node_{self.env_index}")
        self.publisher = self.node.create_publisher(Bool, '/prevent_pause', 10)
        self.client_reset_simulation = self.node.create_client(
            Empty, '/reset_simulation')
        if not self.client_reset_simulation.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error(
                "Service /reset_simulation not available, shutting down node.")
        else:
            self.node.get_logger().info(
                f"Service /reset_simulation is available for env "
                f"{self.env_index}"
            )
            request = Empty.Request()
            future = self.client_reset_simulation.call_async(request)
            rclpy.spin_until_future_complete(self.node, future)

        self._ready.set()
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        try:
            executor.spin()
        except Exception as exc:
            self.node.get_logger().error(
                f"Exception in executor.spin(): {exc}"
            )
        finally:
            executor.shutdown()
            self.node.destroy_node()
            try:
                rclpy.shutdown()
            except RuntimeError:
                self.node.get_logger().info(
                    "rclpy already shutdown, no action needed."
                )

    def wait_until_ready(self, timeout: float = 5.0):
        """Wait until the ROS node and publisher are ready.
        
        Args:
            timeout: Maximum time to wait in seconds.
        """
        self._ready.wait(timeout=timeout)

    def publish(self, value: bool):
        """Publish a boolean value to the /prevent_pause topic.
        
        Args:
            value: Boolean value to publish (True to unpause, False to pause).
        """
        if self.publisher is not None:
            msg = Bool()
            msg.data = value
            self.publisher.publish(msg)
            self.node.get_logger().info(
                f"Env {self.env_index}: Published {value} to /prevent_pause"
            )
        else:
            print(f"Publisher not ready for env {self.env_index}")

        if value is True:
            if self.client_reset_simulation is not None:
                request = Empty.Request()
                future = self.client_reset_simulation.call_async(request)
                rclpy.spin_until_future_complete(self.node, future)
                if future.result() is not None:
                    self.node.get_logger().info(
                        f"Env {self.env_index}: Simulation reset.")
                else:
                    self.node.get_logger().error(
                        f"Env {self.env_index}: Failed to reset simulation.")

class StartStopCallback(BaseCallback):
    """
    Callback that launches one ROS publisher thread per environment.
    Before policy update (_on_rollout_end) it publishes False (pauses)
    and after update (_on_rollout_start) it publishes True (resumes).
    """
    def __init__(self, first_ros_domain_id: int, num_envs: int,
                 update_steps: int, verbose: int = 0):
        """Initialize the callback with ROS publisher threads.
        
        Args:
            first_ros_domain_id: Base ROS domain ID for environment separation.
            num_envs: Number of parallel environments.
            update_steps: Number of steps between policy updates.
            verbose: Verbosity level for callback output.
        """
        super().__init__(verbose)
        self.first_ros_domain_id = first_ros_domain_id
        self.num_envs = num_envs
        self.publisher_threads = {}
        self.update_steps = update_steps
        # Create and start a publisher thread for each environment.
        for i in range(self.num_envs):
            thread = StartStopPublisherThread(i, first_ros_domain_id)
            thread.start()
            thread.wait_until_ready()
            self.publisher_threads[i] = thread

    def _on_rollout_start(self) -> None:
        """Called at the start of a rollout."""
        pass

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        pass

    def _on_step(self) -> bool:
        """Called after each environment step.

        Returns:
            True to continue training, False to stop.
        """
        return True