import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from collections import deque

class SequenceBufferNode(Node):
    def __init__(self):
        super().__init__('sequence_buffer_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/keypoints',
            self.keypoints_callback,
            10)
        self.publisher_ = self.create_publisher(Float32MultiArray, '/sequence', 10)
        
        self.seq_len = 30
        self.buffer = deque(maxlen=self.seq_len)
        
        self.latest_features = None
        
        # Timer to enforce strict 30Hz sequence building
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        
        self.get_logger().info('Sequence Buffer Node started (Sub: /keypoints, Pub: /sequence)')

    def keypoints_callback(self, msg):
        features = np.array(msg.data, dtype=np.float32)
        if len(features) != 84:
            self.get_logger().warning(f'Expected 84 features, got {len(features)}. Ignoring.', throttle_duration_sec=2.0)
            return
            
        self.latest_features = features

    def timer_callback(self):
        # 1. Warm-up phase: do nothing until we get our first valid frame
        if self.latest_features is None:
            return
            
        # 2. Frame drop fallback: we simply append `latest_features` again
        # If the subscriber didn't receive a new message, latest_features remains unchanged.
        self.buffer.append(self.latest_features)
        
        # 3. Publish sequence if buffer is full
        if len(self.buffer) == self.seq_len:
            seq_features = np.array(self.buffer, dtype=np.float32) # (30, 84)
            
            out_msg = Float32MultiArray()
            out_msg.data = seq_features.flatten().tolist()
            self.publisher_.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SequenceBufferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
