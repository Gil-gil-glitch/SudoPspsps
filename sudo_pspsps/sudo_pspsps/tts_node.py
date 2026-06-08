import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from piper import PiperVoice
import io
import wave
import numpy as np
import pyaudio
import json

class SudoTTSNode(Node):
    def __init__(self):
        super().__init__('sudo_tts_node')

        # 1. Initialize local Piper
        model_path = "/home/ri-one/SudoPspsps/src/sudo_pspsps/sudo_pspsps/en_US-amy-low.onnx"
        self.get_logger().info("Loading local Piper voice...")
        self.voice = PiperVoice.load(model_path)
        self.get_logger().info("Piper voice ready ✓")

        # 2. Speaker setup (Piper default is 22050Hz for most models)
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=22050, 
            output=True,
        )

        self.create_subscription(String, '/cat/robot_actions', self._action_callback, 10)
        self.get_logger().info("SudoTTSNode (Offline) ready ✓")

    def _action_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            text = data.get("message_to_user", "").strip()
            if text:
                self._speak(text)
        except Exception as e:
            self.get_logger().error(f"TTS error: {e}")

    def _speak(self, text: str):
        # Generate audio into memory
        audio_buffer = io.BytesIO()
        self.voice.speak(text, audio_buffer)
        
        # Piper outputs WAV data, so we must extract the raw PCM
        audio_buffer.seek(0)
        with wave.open(audio_buffer, 'rb') as wav_file:
            pcm_data = wav_file.readframes(wav_file.getnframes())
            self.stream.write(pcm_data)

    def destroy_node(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SudoTTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()