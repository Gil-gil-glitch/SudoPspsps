import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import base64
import io
import threading

import requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from PIL import Image as PILImage

# ── Server config ─────────────────────────────────────────────────
# Change this to the Mac's IP address on your local network
SERVER_URL = "http://10.42.0.2:8000/infer_vision"


def pil_to_b64(img: PILImage.Image) -> str:
    """Encode a PIL image to a base64 JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class LFMBridgeNode(Node):
    def __init__(self):
        super().__init__('lfm_bridge_node')
        self.bridge = CvBridge()

        # Latest data from each topic
        self.latest_bbox       = None
        self.latest_full_image = None   # PIL image — full frame with bbox drawn
        self.latest_crop_image = None   # PIL image — cropped person

        # --- Subscribers ---
        self.create_subscription(
            Float32MultiArray,
            '/pre_processed_bbox',
            self._bbox_callback,
            10,
        )
        self.create_subscription(
            Image,
            '/pre_processed_tracker_frame',   # Full image with bbox drawn
            self._full_image_callback,
            1,
        )
        self.create_subscription(
            Image,
            '/pre_processed_person_crop',     # Cropped person image — triggers inference
            self._crop_image_callback,
            1,
        )

        # --- Publisher ---
        self.publisher = self.create_publisher(Float32MultiArray, '/cat/vad_state', 10)

        self.get_logger().info(f"LFM bridge ready — sending to {SERVER_URL}")

    # ------------------------------------------------------------------
    # Callbacks — cache the latest value, inference triggered by crop
    # ------------------------------------------------------------------
    def _bbox_callback(self, msg):
        self.latest_bbox = list(msg.data)   # [xmin, ymin, xmax, ymax]

    def _full_image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_full_image = PILImage.fromarray(cv_img)
        except Exception as e:
            self.get_logger().error(f"Full image conversion error: {e}")

    def _crop_image_callback(self, msg):
        """Receives the cropped person image and triggers inference."""
        if self.latest_bbox is None:
            self.get_logger().warn("Waiting for bbox...")
            return
        if self.latest_full_image is None:
            self.get_logger().warn("Waiting for full image...")
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_crop_image = PILImage.fromarray(cv_img)
        except Exception as e:
            self.get_logger().error(f"Crop image conversion error: {e}")
            return

        # Run in a thread so ROS callbacks aren't blocked during the HTTP call
        threading.Thread(target=self._run_inference, daemon=True).start()

    # ------------------------------------------------------------------
    # Inference — just an HTTP POST now, no model on the robot
    # ------------------------------------------------------------------
    def _run_inference(self):
        try:
            payload = {
                "bbox":          self.latest_bbox,
                "full_image_b64": pil_to_b64(self.latest_full_image),
                "crop_image_b64": pil_to_b64(self.latest_crop_image),
            }

            self.get_logger().info(f"Sending request to server (bbox: {self.latest_bbox})...")
            resp = requests.post(SERVER_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            # Server returns parsed floats directly — no parsing needed here
            valence   = float(data["valence"])
            arousal   = float(data["arousal"])
            dominance = float(data["dominance"])

            # Log the raw model output for debugging
            self.get_logger().info(f"Raw response: {data.get('raw', '')}")
            self.get_logger().info(
                f"VAD scores — V:{valence} A:{arousal} D:{dominance}"
            )

            self.publisher.publish(
                Float32MultiArray(data=[valence, arousal, dominance])
            )

        except requests.exceptions.Timeout:
            self.get_logger().error("Request timed out — server may be busy")
        except requests.exceptions.ConnectionError:
            self.get_logger().error(f"Cannot reach server at {SERVER_URL}")
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LFMBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
