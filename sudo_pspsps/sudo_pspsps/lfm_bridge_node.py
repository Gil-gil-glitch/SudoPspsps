import os
# Force every Hugging Face interaction to stay local (must be set before imports)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import torch
import re
import gc
import threading
from unsloth import FastVisionModel
from PIL import Image as PILImage

SYSTEM_PROMPT = (
    "You are an expert emotion analysis assistant for a robot. "
    "Predict Valence, Arousal, and Dominance (VAD) scores (1-10 scale). "
    "You MUST include a rationale for EACH score."
)

# Two images are provided:
#   Image 1 — full scene with a green bounding box around the detected person.
#   Image 2 — close-up crop of just that person.
# The bbox pixel coordinates in the full image are also supplied.
USER_PROMPT = (
    "Two images are provided:\n"
    "  • Image 1: the full scene with a green bounding box at pixel coordinates {bbox}.\n"
    "  • Image 2: a close-up crop of that person.\n\n"
    "Analyze this person's emotion using both images for context.\n"
    "1. Explain the rationale in 2 sentences.\n"
    "2. On a new line for each, output ONLY: 'Valence: [num]', 'Arousal: [num]', 'Dominance: [num]'."
)


class LFMBridgeNode(Node):
    def __init__(self):
        super().__init__('lfm_bridge_node')
        self.bridge = CvBridge()

        # Latest data from each topic
        self.latest_bbox = None
        self.latest_full_image = None   # PIL image — full frame with bbox drawn
        self.latest_crop_image = None   # PIL image — cropped person

        self.model = None
        self.tokenizer = None

        # Load model in background to prevent ROS2 spin freeze
        self.get_logger().info("Initializing node; loading model in background...")
        threading.Thread(target=self._load_model, daemon=True).start()

        # --- Subscribers ---
        self.create_subscription(
            Float32MultiArray,
            '/pre_processed_bbox',
            self._bbox_callback,
            10
        )
        self.create_subscription(
            Image,
            '/pre_processed_tracker_frame',   # Full image with bbox drawn
            self._full_image_callback,
            1
        )
        self.create_subscription(
            Image,
            '/pre_processed_person_crop',     # Cropped person image
            self._crop_image_callback,
            1
        )

        # --- Publisher ---
        self.publisher = self.create_publisher(Float32MultiArray, '/cat/vad_state', 10)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            gc.collect()
            torch.cuda.empty_cache()
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name="/home/ri-one/SudoPspsps/lfm_model_offline",
                max_seq_length=512,
                load_in_4bit=False,
            )
            FastVisionModel.for_inference(self.model)
            self.get_logger().info("Model loaded and ready.")
        except Exception as e:
            self.get_logger().error(f"Model loading failed: {e}")

    # ------------------------------------------------------------------
    # Callbacks — just cache the latest value
    # ------------------------------------------------------------------
    def _bbox_callback(self, msg):
        self.latest_bbox = msg.data  # (xmin, ymin, xmax, ymax)

    def _full_image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_full_image = PILImage.fromarray(cv_img)
        except Exception as e:
            self.get_logger().error(f"Full image conversion error: {e}")

    def _crop_image_callback(self, msg):
        """Receives the cropped person image and triggers inference."""
        if self.model is None:
            self.get_logger().warn("Model not ready yet, skipping frame.")
            return
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

        self._run_inference()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _run_inference(self):
        try:
            bbox_str = (
                f"xmin={int(self.latest_bbox[0])}, ymin={int(self.latest_bbox[1])}, "
                f"xmax={int(self.latest_bbox[2])}, ymax={int(self.latest_bbox[3])}"
            )
            prompt_text = USER_PROMPT.format(bbox=bbox_str)

            # Build message: two images followed by the text prompt
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},   # Image 1 — full scene
                        {"type": "image"},   # Image 2 — person crop
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]

            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            self.get_logger().info(f"Prompt prepared (first 80 chars): {text[:80]}...")

            inputs = self.tokenizer(
                images=[self.latest_full_image, self.latest_crop_image],
                text=[text],
                return_tensors="pt",
            ).to("cuda")

            self.get_logger().info("Running inference...")
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                )

            response = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            self.get_logger().info(f"Raw response: {response}")

            # --- Parse VAD scores ---
            # Look specifically for labelled lines first for robustness
            vad = {}
            for label in ("valence", "arousal", "dominance"):
                match = re.search(rf"{label}:\s*([\d.]+)", response, re.IGNORECASE)
                if match:
                    vad[label] = float(match.group(1))

            if len(vad) == 3:
                scores = [vad["valence"], vad["arousal"], vad["dominance"]]
            else:
                # Fallback: grab first three numbers anywhere in the response
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", response)
                scores = [float(n) for n in nums[:3]] if len(nums) >= 3 else [5.0, 5.0, 5.0]

            self.get_logger().info(f"VAD scores — V:{scores[0]} A:{scores[1]} D:{scores[2]}")
            self.publisher.publish(Float32MultiArray(data=[float(x) for x in scores]))

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
