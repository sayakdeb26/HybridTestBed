import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np

class KeypointExtractorNode(Node):
    def __init__(self):
        super().__init__('keypoint_extractor_node')
        self.subscription = self.create_subscription(
            Image,
            '/image',
            self.image_callback,
            10)
        self.publisher_ = self.create_publisher(Float32MultiArray, '/keypoints', 10)
        self.bridge = CvBridge()
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
            
        self.pose_indices = [11, 12, 13, 14, 15, 16] # Shoulders, Elbows, Wrists
        self.hand_indices = [0, 4, 8, 12, 16, 20] # Wrist, Thumb, Index, Middle, Ring, Pinky
        
        # Temporal buffers for missing keypoints and velocity
        self.last_valid_kps = {
            'pose': None,
            'left': None,
            'right': None
        }
        self.prev_base_features = None
        self.vel_indices = list(range(12, 18)) + list(range(21, 30)) + list(range(39, 48))
        
        self.get_logger().info('Keypoint Extractor Node started (Sub: /image, Pub: /keypoints)')

    def extract_features(self, results):
        features = []
        
        # Fallback to last valid if missing
        pose_kps = results.pose_landmarks.landmark if results.pose_landmarks else self.last_valid_kps['pose']
        left_kps = results.left_hand_landmarks.landmark if results.left_hand_landmarks else self.last_valid_kps['left']
        right_kps = results.right_hand_landmarks.landmark if results.right_hand_landmarks else self.last_valid_kps['right']
        
        # Update buffers
        if pose_kps: self.last_valid_kps['pose'] = pose_kps
        if left_kps: self.last_valid_kps['left'] = left_kps
        if right_kps: self.last_valid_kps['right'] = right_kps

        # 1. Setup origin (Shoulder midpoint)
        if pose_kps:
            l_shoulder = np.array([pose_kps[11].x, pose_kps[11].y, pose_kps[11].z])
            r_shoulder = np.array([pose_kps[12].x, pose_kps[12].y, pose_kps[12].z])
            origin = (l_shoulder + r_shoulder) / 2.0
        else:
            origin = np.zeros(3)

        # 2. Pose (18 dims)
        if pose_kps:
            for idx in self.pose_indices:
                pt = np.array([pose_kps[idx].x, pose_kps[idx].y, pose_kps[idx].z]) - origin
                features.extend(pt.tolist())
        else:
            features.extend([0.0] * 18)

        # 3. Hands (36 dims)
        for hand_kps in [left_kps, right_kps]:
            if hand_kps:
                for idx in self.hand_indices:
                    pt = np.array([hand_kps[idx].x, hand_kps[idx].y, hand_kps[idx].z]) - origin
                    features.extend(pt.tolist())
            else:
                features.extend([0.0] * 18)

        # 4. Motion (6 dims) - wrist displacement
        if pose_kps:
            l_wrist = np.array([pose_kps[15].x, pose_kps[15].y, pose_kps[15].z]) - origin
            r_wrist = np.array([pose_kps[16].x, pose_kps[16].y, pose_kps[16].z]) - origin
            features.extend(l_wrist.tolist())
            features.extend(r_wrist.tolist())
        else:
            features.extend([0.0] * 6)

        base_features = np.array(features, dtype=np.float32)
        
        # 5. Velocity (24 dims)
        velocities = np.zeros(24, dtype=np.float32)
        if self.prev_base_features is not None:
            velocities = base_features[self.vel_indices] - self.prev_base_features[self.vel_indices]
            
        self.prev_base_features = base_features
        
        # 6. Final 84-dim vector
        final_vector = np.concatenate([base_features, velocities])
        return final_vector

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            results = self.holistic.process(image_rgb)
            features = self.extract_features(results)
            
            feature_msg = Float32MultiArray()
            feature_msg.data = features.tolist()
            self.publisher_.publish(feature_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = KeypointExtractorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
