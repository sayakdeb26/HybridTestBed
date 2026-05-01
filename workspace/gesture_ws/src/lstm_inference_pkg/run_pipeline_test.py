import os
import numpy as np
import onnxruntime as ort
import json
import time

# Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(project_root, 'hand_gesture_lab', 'data', 'processed', 'train')
X_path = os.path.join(data_dir, 'X_fixed.npy')
y_path = os.path.join(data_dir, 'y.npy')
onnx_path = os.path.join(project_root, 'hand_gesture_lab', 'weights', 'best_lstm_model.onnx')

# Load dataset (already pre‑processed to feature vectors)
X = np.load(X_path)  # shape (N, T, F)
y = np.load(y_path)
print(f'Loaded dataset: {X.shape[0]} samples, sequence length {X.shape[1]}, feature dim {X.shape[2]}')

# Load ONNX model
if not os.path.exists(onnx_path):
    raise FileNotFoundError(f'ONNX model not found at {onnx_path}')
sess = ort.InferenceSession(onnx_path)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Simple rolling buffer – keep the last T frames (already full length)
# For testing we will feed a few consecutive samples and inspect the predictions.
buffer_size = X.shape[1]

# Helper to run inference on a single sequence
def infer(sequence):
    # sequence shape (T, F)
    tensor = sequence[np.newaxis, :, :].astype(np.float32)
    # Measure inference latency
    start_time = time.time()
    logits = sess.run([output_name], {input_name: tensor})[0]  # (1, num_classes)
    elapsed_ms = (time.time() - start_time) * 1000.0
    # Store latency for later averaging
    if not hasattr(infer, "latencies"):
        infer.latencies = []
    infer.latencies.append(elapsed_ms)
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    probs = probs.squeeze(0)
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return pred_idx, confidence

# Run through first 30 samples and collect predictions
pred_buffer = []
stable = True
prev_pred = None
for i in range(30):
    seq = X[i]
    pred, conf = infer(seq)
    pred_buffer.append((pred, conf))
    # Simple stability check: if confidence < 0.75 flag potential flicker
    if conf < 0.75:
        stable = False
    # Check for sudden label change when confidence is high
    if prev_pred is not None and pred != prev_pred and conf > 0.9:
        print(f'⚠️ Sudden high‑confidence change at sample {i}: {prev_pred} → {pred}')
    prev_pred = pred
    print(f'Sample {i:02d}: Pred={pred} ({conf:.3f})')

print('\n=== Summary ===')
print(f'Stable (all confidences ≥0.75): {stable}')
print(f'Buffer length: {len(pred_buffer)} (last 5 will be used by the ROS node)')
if hasattr(infer, 'latencies'):
    avg_latency = np.mean(infer.latencies)
    print(f'Average inference latency: {avg_latency:.2f} ms')

# Show what the ROS node would publish for the last buffer entry
last_five = pred_buffer[-5:]
ids, confs = zip(*last_five)
# Majority vote
majority_id = max(set(ids), key=ids.count)
mean_conf = np.mean([c for (i, c) in last_five if i == majority_id])
label_map = ["Swipe Left", "Swipe Right", "Swipe Up", "Swipe Down", "Push Away"]
label = label_map[majority_id] if mean_conf >= 0.75 else "UNCERTAIN"
print(f'ROS publish simulation -> label: {label}, confidence: {mean_conf:.3f}')
