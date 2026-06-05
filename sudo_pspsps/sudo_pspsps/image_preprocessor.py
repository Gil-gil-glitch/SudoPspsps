import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import mediapipe as mp


class ImagePreprocessor(Node):
    def __init__(self):
        super().__init__('image_preprocessor')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            10
        )
        # Publisher 1: Full image with bounding box drawn on it (spatial context)
        self.full_image_publisher = self.create_publisher(Image, '/pre_processed_tracker_frame', 10)
        # Publisher 2: Cropped image of just the person (detail context)
        self.crop_image_publisher = self.create_publisher(Image, '/pre_processed_person_crop', 10)
        # Publisher 3: Bounding box coordinates [xmin, ymin, xmax, ymax] in original image space
        self.bbox_publisher = self.create_publisher(Float32MultiArray, '/pre_processed_bbox', 10)

        self.bridge = CvBridge()
        self.frame_counter = 0

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5
        )

    def listener_callback(self, msg):
        """
        Process incoming images, detect pose landmarks, and publish:
          1. Full image (336x336) with bounding box drawn — gives LFM spatial context.
          2. Cropped person image (224x224) — gives LFM close-up detail.
          3. Bounding box coordinates in original image pixel space.
        """
        self.frame_counter += 1
        if self.frame_counter % 200 != 0:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if not results.pose_landmarks:
            self.get_logger().info("No person detected, skipping frame.")
            return

        h, w, _ = cv_image.shape

        # --- Compute bounding box from pose landmarks ---
        x_coords = [int(lm.x * w) for lm in results.pose_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in results.pose_landmarks.landmark]
        xmin = max(0, min(x_coords) - 20)
        xmax = min(w, max(x_coords) + 20)
        ymin = max(0, min(y_coords) - 20)
        ymax = min(h, max(y_coords) + 20)

        # --- Publish bbox coordinates (in original image space) ---
        bbox_msg = Float32MultiArray()
        bbox_msg.data = [float(xmin), float(ymin), float(xmax), float(ymax)]
        self.bbox_publisher.publish(bbox_msg)

        # --- Full image: draw bounding box, then resize to 336x336 ---
        annotated = cv_image.copy()
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Optional: local debug window
        cv2.imshow("Detection Preview", annotated)
        cv2.waitKey(1)

        full_resized = cv2.resize(annotated, (336, 336))
        full_msg = self.bridge.cv2_to_imgmsg(full_resized, encoding='bgr8')
        full_msg.header = msg.header  # Preserve timestamp for synchronization
        self.full_image_publisher.publish(full_msg)

        # --- Cropped person image: slice bbox region, resize to 224x224 ---
        # Guard against degenerate boxes
        if xmax > xmin and ymax > ymin:
            person_crop = cv_image[ymin:ymax, xmin:xmax]
            crop_resized = cv2.resize(person_crop, (224, 224))
            crop_msg = self.bridge.cv2_to_imgmsg(crop_resized, encoding='bgr8')
            crop_msg.header = msg.header
            self.crop_image_publisher.publish(crop_msg)
        else:
            self.get_logger().warn("Degenerate bounding box, skipping crop publish.")


def main(args=None):
    rclpy.init(args=args)
    node = ImagePreprocessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
