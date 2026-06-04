import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePreprocessor(Node):
    def __init__(self):
        super().__init__('image_preprocessor')
        self.subscription = self.create_subscription(Image, '/camera/camera/color/image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher(Image, '/pre_processed_tracker_frame', 10)
        self.bridge = CvBridge()
        self.timer_counter = 0

    def listener_callback(self, msg):
        # Throttle: process only every 10th frame (assuming 30fps = 3fps processing)
        self.timer_counter += 1
        if self.timer_counter % 200 != 0:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Resize for LFM-VL model input
        resized_image = cv2.resize(cv_image, (336, 336))
        
        # Convert back and publish
        out_msg = self.bridge.cv2_to_imgmsg(resized_image, encoding='bgr8')
        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePreprocessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()