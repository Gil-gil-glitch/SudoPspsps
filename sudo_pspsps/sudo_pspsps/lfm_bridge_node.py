import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import torch
import re
from unsloth import FastVisionModel
from PIL import Image as PILImage

class LFMBridgeNode(Node):
    def __init__(self):
        super().__init__('lfm_bridge_node')
        self.bridge = CvBridge()
        
        # 1. Load Model
        self.get_logger().info("Loading LFM model...")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name="LiquidAI/LFM2.5-VL-1.6B",
            max_seq_length=2048,
            load_in_4bit=False,
        )
        FastVisionModel.for_inference(self.model)
        self.get_logger().info("Model loaded and ready.")

        # 2. Setup Subscriber and Publisher
        self.subscriber = self.create_subscription(
            Image, '/pre_processed_tracker_frame', self.vision_callback, 10)
        self.publisher = self.create_publisher(Float32MultiArray, '/cat/vad_state', 10)

        # 3. Prompt setup: "Chain of Thought" forces rationale
        self.system_prompt = (
            "You are an emotion analysis assistant. "
            "STEP 1: Explain the person's facial expressions and body language in 2-3 sentences. "
            "STEP 2: Based on that, determine the VAD scores (1-10 scale). "
            "Output format:\nRationale: <explanation>\nValence: <num>\nArousal: <num>\nDominance: <num>"
        )

    def parse_vad(self, text):
        # This regex ignores case, handles optional colons, and matches numbers anywhere
        def extract(label, text):
            # Matches "Label: 5" or "Label 5" or "Label : 5.0"
            pattern = rf"{label}\s*[:\-]?\s*(\d+(?:\.\d+)?)"
            match = re.search(pattern, text, re.IGNORECASE)
            return float(match.group(1)) if match else None

        v = extract("Valence", text)
        a = extract("Arousal", text)
        d = extract("Dominance", text)
        
        if v is not None and a is not None and d is not None:
            vals = [v, a, d]
            # If the model output a normalized score (0.0 to 1.0), scale it
            cleaned = [x * 10 if x < 1.0 else x for x in vals]
            # Clamp between 1.0 and 10.0
            return [max(1.0, min(10.0, x)) for x in cleaned]
        
        self.get_logger().warn("Regex failed to find VAD values. Returning default.")
        return [5.0, 5.0, 5.0]

    def vision_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            pil_image = PILImage.fromarray(cv_image)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Analyze the human's emotional state."}]}
            ]
            
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.tokenizer(images=[pil_image], text=[text], return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                # Increased tokens to ensure rationale fits
                out = self.model.generate(**inputs, max_new_tokens=300, temperature=0.5, do_sample=True) 
            
            full_output = self.tokenizer.decode(out[0], skip_special_tokens=True)
            
            # Isolate the model's rationale response
            assistant_response = full_output.split("assistant")[-1].strip()
            
            # Print verbose debugging to terminal
            self.get_logger().info(f"\n--- AI REASONING ---\n{assistant_response}\n--------------------")
            
            # Extract scores
            v, a, d = self.parse_vad(assistant_response)

            vad_msg = Float32MultiArray()
            vad_msg.data = [v, a, d]
            self.publisher.publish(vad_msg)
            self.get_logger().info(f"Published VAD: V={v:.1f}, A={a:.1f}, D={d:.1f}")
            
        except Exception as e:
            self.get_logger().error(f"Error during inference: {e}")

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