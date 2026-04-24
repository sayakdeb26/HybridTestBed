import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import os
import csv
import json

class DatasetSourceNode(Node):
    def __init__(self):
        super().__init__('dataset_source_node')
        
        # Declare parameters
        self.declare_parameter('video_dir', '/home/sayak/datasets/jester/Train')
        self.declare_parameter('csv_path', '/home/sayak/datasets/jester/Train.csv')
        self.declare_parameter('fps', 30.0)
        
        self.video_dir = self.get_parameter('video_dir').get_parameter_value().string_value
        self.csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value
        
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.meta_pub = self.create_publisher(String, '/video_meta', 10)
        
        self.bridge = CvBridge()
        
        # Load dataset
        self.vid_to_label = self._load_csv()
        self.video_ids = sorted(list(self.vid_to_label.keys()))
        self.current_video_idx = 0
        
        self.current_frames = []
        self.current_frame_idx = 0
        
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        self.get_logger().info(f'Dataset Source Node started. Found {len(self.video_ids)} videos.')
        
        self._load_next_video()

    def _load_csv(self):
        vid_to_label = {}
        if not os.path.exists(self.csv_path):
            self.get_logger().error(f'CSV not found at {self.csv_path}')
            return vid_to_label
            
        with open(self.csv_path, 'r') as f:
            # Jester CSV uses semicolon or comma
            sample = f.readline()
            sep = ';' if ';' in sample else ','
            f.seek(0)
            
            reader = csv.reader(f, delimiter=sep)
            for row in reader:
                if len(row) >= 2:
                    vid_id = row[0].strip()
                    label = row[1].strip()
                    if vid_id != 'video_id': # skip header
                        vid_to_label[vid_id] = label
        return vid_to_label

    def _load_next_video(self):
        if self.current_video_idx >= len(self.video_ids):
            self.get_logger().info('End of dataset reached.')
            self.current_frames = []
            return
            
        vid_id = self.video_ids[self.current_video_idx]
        label = self.vid_to_label[vid_id]
        
        vid_path = os.path.join(self.video_dir, str(vid_id))
        
        if os.path.isdir(vid_path):
            # Sort frames by name (assuming format like 00001.jpg)
            frames = sorted([f for f in os.listdir(vid_path) if f.endswith('.jpg') or f.endswith('.png')])
            self.current_frames = [os.path.join(vid_path, f) for f in frames]
            self.current_frame_idx = 0
            
            # Publish metadata at the start of the video
            meta_msg = String()
            meta_msg.data = json.dumps({"video_id": vid_id, "label": label})
            self.meta_pub.publish(meta_msg)
            
            self.get_logger().info(f'Playing video {vid_id} ({label}) - {len(self.current_frames)} frames')
        else:
            self.get_logger().warning(f'Video directory not found: {vid_path}')
            self.current_video_idx += 1
            self._load_next_video()

    def timer_callback(self):
        if not self.current_frames:
            return
            
        if self.current_frame_idx < len(self.current_frames):
            frame_path = self.current_frames[self.current_frame_idx]
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.image_pub.publish(msg)
            else:
                self.get_logger().error(f'Failed to read frame: {frame_path}')
                
            self.current_frame_idx += 1
        else:
            # Video finished
            self.current_video_idx += 1
            self._load_next_video()

def main(args=None):
    rclpy.init(args=args)
    node = DatasetSourceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
