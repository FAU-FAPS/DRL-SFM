"""Module for monitoring simulation clock and managing pause/unpause
functionality."""


import argparse
import os
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Bool

class ClockMonitor(Node):
    """ROS node for monitoring simulation time and controlling pause state."""
    
    def __init__(self):
        """Initialize the clock monitor node."""
        # Use a generic node name; the ROS_DOMAIN_ID is now set in main().
        super().__init__('clock_monitor')
        
        # Create a QoS profile for the clock subscription.
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to the /clock topic.
        self.clock_sub = self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            qos_profile
        )
        
        # Subscribe to the /prevent_pause topic, wich contains a Bool message.
        self.prevent_sub = self.create_subscription(
            Bool,
            '/prevent_pause',
            self.prevent_pause_callback,
            qos_profile
        )

        self.client_unpause = self.create_client(Empty, '/unpause_physics')
        self.client_pause = self.create_client(Empty, '/pause_physics')
        self.last_received_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.check_timeout)
        self.prevent_pause = True
        self.pause_counter = 0
    
    def clock_callback(self, msg):
        """Callback function for clock subscription."""
        # Update last received time on each new clock message.
        self.last_received_time = self.get_clock().now()
    
    def prevent_pause_callback(self, msg):
        """Callback function for prevent_pause subscription."""
        self.prevent_pause = msg.data
        if self.prevent_pause == False:
            self.get_logger().warn(
                'Prevent pause flag set to False, '
                'pausing physics...'
            )
            self.call_pause_physics()

    def check_timeout(self):
        """Check for clock timeouts and manage physics pause state."""
        elapsed_time = (
            self.get_clock().now() - self.last_received_time
        ).nanoseconds / 1e9
        if elapsed_time > 1.0:
            if self.prevent_pause:
                self.get_logger().warn(
                    'No clock message received for 1s, '
                    'unpausing physics...'
                )
                self.call_unpause_physics()
                self.pause_counter = 0
            else:
                self.pause_counter += 1
                if self.pause_counter % 10 == 0:
                    self.get_logger().warn(
                        f'Training for {self.pause_counter // 10}s'
                    )

    def call_unpause_physics(self):
        """Call the /unpause_physics service."""
        if not self.client_unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Service /unpause_physics not available')
            return
        
        # Create and send an empty request.
        request = Empty.Request()
        self.future = self.client_unpause.call_async(request)
        self.future.add_done_callback(self.response_callback_unpause)
    
    def response_callback_unpause(self, future):
        """Response callback for the /unpause_physics service call."""
        try:
            future.result()
            self.get_logger().info(
                'Successfully called /unpause_physics service'
            )
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def call_pause_physics(self):
        """Call the /pause_physics service."""
        if not self.client_pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Service /pause_physics not available')
            return
        
        # Create and send an empty request.
        request = Empty.Request()
        self.future = self.client_pause.call_async(request)
        self.future.add_done_callback(self.response_callback_pause)

    def response_callback_pause(self, future):
        """Response callback for the /pause_physics service call."""
        try:
            future.result()
            self.get_logger().info(
                'Successfully called '
                '/pause_physics service'
            )
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')
    

def main(args=None):
    """Main function to parse arguments and start the ROS node."""
    # Parse command-line arguments to get the ROS domain ID.
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=int, default=10,
                        help="ROS Domain ID for ClockMonitor")
    parsed_args, _ = parser.parse_known_args()
    
    # Set the ROS_DOMAIN_ID before initializing rclpy.
    os.environ["ROS_DOMAIN_ID"] = str(parsed_args.domain)
    
    rclpy.init(args=args)
    node = ClockMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
