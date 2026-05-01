import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import onnxruntime as ort
import os

# Label map matching the training classes
LABEL_MAP = ["Swipe Left", "Swipe Right", "Swipe Up", "Swipe Down", "Push Away"]

class LSTMInferenceNode(Node):
    def __init__(self):
        super().__init__('lstm_inference_node')
        self.pred_buffer = []  # buffer for last 5 predictions
        # Parameters (could be overridden via ROS2 param system)
        self.declare_parameter('onnx_path', 'hand_gesture_lab/weights/best_lstm_model.onnx')
        onnx_path = self.get_parameter('onnx_path').get_parameter_value().string_value
        # Resolve absolute path
        if not os.path.isabs(onnx_path):
            # Assume workspace root as base
            pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            onnx_path = os.path.join(pkg_dir, onnx_path)
        if not os.path.exists(onnx_path):
            self.get_logger().error(f"ONNX model not found at {onnx_path}")
            raise FileNotFoundError(onnx_path)
        self.get_logger().info(f"Loading ONNX model from {onnx_path}")
        self.session = ort.InferenceSession(onnx_path)
        # Input name is inferred from the model
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Subscriber to sequence (expects Float32MultiArray with shape 30 x F)
        self.sub = self.create_subscription(
            Float32MultiArray,
            '/sequence',
            self.sequence_callback,
            10)
        # Publisher for prediction
        self.pub = self.create_publisher(String, '/prediction', 10)
        self.get_logger().info('LSTM inference node started')

    def sequence_callback(self, msg: Float32MultiArray):
        # Convert incoming data to NumPy array. Expect layout: data list flat; we reshape to (30, F)
        data = np.array(msg.data, dtype=np.float32)
        # Infer feature dimension from model input shape (batch, T, F)
        # The model expects shape (1, T, F)
        try:
            # Guess feature dim from data length
            T = 30  # fixed sequence length
            F = data.size // T
            if data.size % T != 0:
                self.get_logger().warning('Received sequence length not divisible by 30, ignoring')
                return
            input_tensor = data.reshape(1, T, F)
        except Exception as e:
            self.get_logger().error(f'Failed to reshape input: {e}')
            return

        # Run ONNX inference
        ort_inputs = {self.input_name: input_tensor}
        try:
            ort_outs = self.session.run([self.output_name], ort_inputs)
        except Exception as e:
            self.get_logger().error(f'ONNX inference error: {e}')
            return
        logits = ort_outs[0]  # shape (1, num_classes)
        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        probs = probs.squeeze(0)  # (num_classes,)

        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx])

        # If confidence too low, mark as uncertain
        if best_conf < 0.5:
            self.pred_buffer.append((-1, best_conf))
        else:
            self.pred_buffer.append((best_idx, best_conf))

        if len(self.pred_buffer) > 5:
            self.pred_buffer.pop(0)

        # Majority vote
        idxs = [p[0] for p in self.pred_buffer]
        counts = np.bincount(idxs, minlength=len(LABEL_MAP) + 1)
        smooth_idx = int(np.argmax(counts))
        
        # Output logic
        if smooth_idx == -1 or counts[smooth_idx] < 3:
            label_name = "Uncertain"
            publish_conf = 0.0
            label_val = -1
        else:
            confs = [p[1] for p in self.pred_buffer if p[0] == smooth_idx]
            publish_conf = float(np.mean(confs))
            label_name = LABEL_MAP[smooth_idx]
            label_val = smooth_idx
        
        pred_msg = String()
        pred_msg.data = json.dumps({"label": label_name, "confidence": publish_conf})
        self.pub.publish(pred_msg)
        self.get_logger().info(f'Predicted (smoothed): {label_name} confidence={publish_conf:.3f}')

def main(args=None):
    rclpy.init(args=args)
    node = LSTMInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
