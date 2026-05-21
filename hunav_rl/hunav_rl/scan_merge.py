"""Module for merging multiple laser scan topics into a single unified scan."""


from sensor_msgs.msg import LaserScan
from rclpy.node import Node
import rclpy
import numpy as np
import time


class ScanMerge(Node):
    """ROS node for merging visual and collision laser scans."""
    
    def __init__(self):
        """Initialize the scan merge node."""
        super().__init__('scan_merge')

        # Publisher for merged scan
        self.merged_scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        # Subscribers for input scans, queue size set to 1
        self.create_subscription(LaserScan, '/scan_visual',
                     self.scan_visual_callback, 1)
        self.create_subscription(LaserScan, '/scan_collision',
                     self.scan_collision_callback, 1)

        # Store the latest scans
        self.scan_visual = None
        self.scan_visual_old = None
        self.scan_collision = None
        self.scan_collision_old = None

        # Timer to merge at 30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        # Track timestamps to detect new messages
        self.last_visual_time = None
        self.last_collision_time = None
        self.received_scans = False

        self.get_logger().info('ScanMerge node initialized at 30 Hz')

    def scan_visual_callback(self, msg):
        """Callback for visual scan topic."""
        # check for same message if gazebo is paused/reset
        if self.scan_visual_old is not None:
            old_stamp = (
                self.scan_visual_old.header.stamp.sec,
                self.scan_visual_old.header.stamp.nanosec,
            )
            new_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            if old_stamp == new_stamp:
                return 

        self.scan_visual = msg
        self.last_visual_time = time.time()

    def scan_collision_callback(self, msg):
        """Callback for collision scan topic."""
        # check for same message if gazebo is paused/reset
        if self.scan_collision_old is not None:
            old_stamp = (
                self.scan_collision_old.header.stamp.sec,
                self.scan_collision_old.header.stamp.nanosec,
            )
            new_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            if old_stamp == new_stamp:
                return 


        self.scan_collision = msg
        self.last_collision_time = time.time()

    def timer_callback(self):
        """Timer callback to merge scans at regular intervals."""
        # Check if we have received both scans
        if (not self.received_scans and self.scan_visual is not None
            and self.scan_collision is not None):
            self.received_scans = True

        # Only merge if both scans are available and have been updated
        if self.scan_visual is None or self.scan_collision is None:
            return

        # Merge using latest scans
        self.merge_and_publish()

    def merge_and_publish(self):
        """Merge the latest visual and collision scans."""
        visual_time = (
            self.scan_visual.header.stamp.sec
            + self.scan_visual.header.stamp.nanosec * 1e-9
        )
        collision_time = (
            self.scan_collision.header.stamp.sec
            + self.scan_collision.header.stamp.nanosec * 1e-9
        )

        if visual_time > collision_time:
            base_scan = self.scan_visual
            other_scan = self.scan_collision
        else:
            base_scan = self.scan_collision
            other_scan = self.scan_visual

        # Check if scans have compatible parameters
        if (len(base_scan.ranges) != len(other_scan.ranges) or
            abs(base_scan.angle_min - other_scan.angle_min) > 1e-6 or
            abs(base_scan.angle_max - other_scan.angle_max) > 1e-6 or
            abs(base_scan.angle_increment -
                other_scan.angle_increment) > 1e-6):
            self.get_logger().warn(
            'Incompatible scan parameters, skipping merge'
            )
            return

        merged_scan = LaserScan()
        merged_scan.header = base_scan.header
        merged_scan.angle_min = base_scan.angle_min
        merged_scan.angle_max = base_scan.angle_max
        merged_scan.angle_increment = base_scan.angle_increment
        merged_scan.time_increment = base_scan.time_increment
        merged_scan.scan_time = base_scan.scan_time
        merged_scan.range_min = min(base_scan.range_min, other_scan.range_min)
        merged_scan.range_max = max(base_scan.range_max, other_scan.range_max)

        ranges1 = np.array(base_scan.ranges, dtype=np.float32)
        ranges2 = np.array(other_scan.ranges, dtype=np.float32)

        merged_ranges = np.minimum(ranges1, ranges2)
        merged_scan.ranges = merged_ranges.tolist()

        # Merge intensities if both scans have them
        if (hasattr(base_scan, 'intensities') and
            hasattr(other_scan, 'intensities') and
            len(base_scan.intensities) == len(other_scan.intensities)):
            intensities1 = np.array(base_scan.intensities, dtype=np.float32)
            intensities2 = np.array(other_scan.intensities, dtype=np.float32)
            mask = ranges1 <= ranges2
            merged_intensities = np.where(mask, intensities1, intensities2)
            merged_scan.intensities = merged_intensities.tolist()

        self.merged_scan_pub.publish(merged_scan)
        
        self.scan_visual_old = self.scan_visual
        self.scan_collision_old = self.scan_collision
        self.scan_visual = None
        self.scan_collision = None


def main(args=None):
    rclpy.init(args=args)
    node = ScanMerge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
