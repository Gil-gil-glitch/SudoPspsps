import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import torch
import re
from unsloth import FastVisionModel
from PIL import Image as PILImage
 
 
SYSTEM_PROMPT = (
    "You are an emotion analysis assistant for a companion robot. "
    "Given an image of a person, predict their Valence, Arousal, and Dominance (VAD) scores. "
    "Valence measures how positive/negative they feel (1=very negative, 10=very positive). "
    "Arousal measures their energy/activation level (1=very calm/sleepy, 10=very excited/agitated). "
    "Dominance measures their sense of control (1=feeling helpless/controlled, 10=feeling powerful/in control). "
    "You may also optionally note any visible EMOTIC emotion categories. "
    "Always respond in this exact format:\n"
    "Rationale: <2-3 sentence explanation of body language and facial expression>\n"
    "Valence: <number 1-10>\n"
    "Arousal: <number 1-10>\n"
    "Dominance: <number 1-10>\n"
    "Emotions: <comma separated list or None>"
)
 
USER_PROMPT = (
    "Analyse this person's emotional state based on body language, "
    "posture, facial expression, and surrounding context. "
    "Give VAD scores and optionally note any emotion categories you observe."
)
 
 
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
 
    def parse_rationale(self, text):
        match = re.search(r"Rationale\s*[:\-]?\s*(.+?)(?=\nValence|\nArousal|\nDominance|$)",
                          text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else "No rationale found."
 
    def parse_vad(self, text):
        def extract(label, text):
            pattern = rf"{label}\s*[:\-]?\s*(\d+(?:\.\d+)?)"
            match = re.search(pattern, text, re.IGNORECASE)
            return float(match.group(1)) if match else None
 
        v = extract("Valence", text)
        a = extract("Arousal", text)
        d = extract("Dominance", text)
 
        if v is not None and a is not None and d is not None:
            vals = [v, a, d]
            # Scale up if model returned normalized 0.0-1.0 values
            cleaned = [x * 10 if x <= 1.0 else x for x in vals]
            # Clamp to 1.0-10.0
            return [max(1.0, min(10.0, x)) for x in cleaned]
 
        self.get_logger().warn(
            f"Failed to parse VAD from response. Returning default [5,5,5].\n"
            f"Response was:\n{text}"
        )
        return [5.0, 5.0, 5.0]
 
    def vision_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            pil_image = PILImage.fromarray(cv_image)
 
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT}
                ]}
            ]
 
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
            inputs = self.tokenizer(
                images=[pil_image], text=[text], return_tensors="pt").to("cuda")
 
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.1,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                )
 
            # ---- KEY FIX: slice off the input tokens, decode only new output ----
            input_len = inputs["input_ids"].shape[1]
            generated_ids = out[0][input_len:]
            assistant_response = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True).strip()
            # ---------------------------------------------------------------------
 
            rationale = self.parse_rationale(assistant_response)
            self.get_logger().info(f"\n--- AI REASONING ---\n{rationale}\n--------------------")
 
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
