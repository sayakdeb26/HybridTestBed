import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import numpy as np
import json
import random

class LSTMInferenceNode(Node):
    def __init__(self):
        super().__init__('lstm_inference_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/sequence',
            self.sequence_callback,
            10)
        self.publisher_ = self.create_publisher(String, '/prediction', 10)
        
        self.get_logger().info('LSTM Inference Node started (Sub: /sequence, Pub: /prediction)')

    def sequence_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) != 2520:
            self.get_logger().warning(f'Expected 2520 elements (30x84), got {len(data)}. Ignoring.', throttle_duration_sec=2.0)
            return
            
        sequence = data.reshape((1, 30, 84))
        
        # TODO: Replace with actual PyTorch model forward pass
        # output = self.model(torch.tensor(sequence).to(self.device))
        
        # Mock prediction
        prediction = {
            "label": "test",
            "confidence": round(random.uniform(0.5, 0.99), 2)
        }
        
        out_msg = String()
        out_msg.data = json.dumps(prediction)
        self.publisher_.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LSTMInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
