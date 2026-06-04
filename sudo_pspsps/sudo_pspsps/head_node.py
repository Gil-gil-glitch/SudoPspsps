#
## head_node.py
#
#  ROS2 node for controlling the pan-tilt mechanism on the cat head. This node
#  subscribes to target pan and tilt angles from a tracking node and smoothly 
#  moves the head towards those targets using the DynamixelDriver. This node
#  acts as the interface between ROS2 and the Dynamixel hardware.
#

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from .dynamixel_driver import DynamixelDriver


class HeadNode(Node):

    def __init__(self):
        super().__init__('head_node')

        # Dynamixel Driver Initialization
        self.driver = DynamixelDriver()

        self.driver.calibrate_zero()
        self.driver.enable()

        # Current and Target States
        self.current_pan = 0.0
        self.current_tilt = 0.0

        self.target_pan = 0.0
        self.target_tilt = 0.0

        # Subscriptions
        self.create_subscription(
            Float32,
            '/head/pan_target',
            self.pan_callback,
            10
        )


        self.create_subscription(
            Float32,
            '/head/tilt_target',
            self.tilt_callback,
            10
        )

        # Control Loop Timer
        self.timer = self.create_timer(
            0.05,   # 20 Hz
            self.control_loop
        )

        self.get_logger().info(
            "Head node started"
        )

    # Topic Callbacks
    def pan_callback(self, msg):

        self.target_pan = msg.data

    def tilt_callback(self, msg):

        self.target_tilt = msg.data

    # Main Control Loop
    def control_loop(self):

        alpha = 0.15

        self.current_pan += (
            self.target_pan -
            self.current_pan
        ) * alpha

        self.current_tilt += (
            self.target_tilt -
            self.current_tilt
        ) * alpha

        self.driver.set_pan(
            self.current_pan
        )

        self.driver.set_tilt(
            self.current_tilt
        )

    # Turn off motors and close driver on shutdown
    def destroy_node(self):

        self.driver.disable()
        self.driver.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = HeadNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()