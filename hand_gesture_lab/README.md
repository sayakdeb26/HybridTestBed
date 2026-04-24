# Real-Time Hand Gesture Recognition

A modular, robust prototype for real-time hand gesture recognition using MediaPipe and PyTorch LSTMs. Designed to be seamlessly integrated into ROS2 pipelines.

## 🚀 Recent Pipeline Upgrades
The classification engine has been heavily optimized from a 27-class baseline to a highly accurate **5-class motion gesture system**:
- **Target Gestures**: `Swipe Left`, `Swipe Right`, `Swipe Up`, `Swipe Down`, and `Push Away`.
- **Advanced Feature Engineering**: The feature vector has been upgraded to a 191-dimensional space including:
  - Raw un-normalized coordinates relative to the shoulder midpoint.
  - Temporal velocity and acceleration over all joints.
  - Overall motion magnitude (`||velocity||`).
  - Wrist-specific direction encoding normalized vectors.
  - **Body-Relative Arm Extension**: Scale-invariant wrist-to-shoulder depth computation to reliably isolate "Push Away" gestures.
- **Model Architecture**: Uses a Bidirectional LSTM (96 -> 48) combined with temporal logit smoothing (`alpha=0.6`) over the sequence axis to simulate streaming inference and stabilize real-time predictions.
- **Data Augmentation**: Employs Gaussian noise injection, time warping, and random feature masking during training to drastically reduce overfitting.

## Features
- **Two-Stage Detection**: Uses MediaPipe Pose to locate shoulders and wrists, creating a dynamic Region of Interest (ROI) for MediaPipe Hands. This significantly improves accuracy and reduces false positives.
- **Temporal Logit Smoothing**: Replaces static context attention with frame-by-frame evaluation followed by exponential smoothing to ensure highly stable classifications during live streaming.
- **Carry-Forward Imputation**: Automatically recovers dropped frames/missing hand detections to prevent velocity spikes.
- **BiLSTM Sequence Classifier**: Uses a fast, heavily regularized 2-layer Bidirectional LSTM for classifying 30-frame keypoint sequences.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training on the Jester Dataset

### 1. Data Preprocessing
Run the preprocessor to extract 30-frame `.npy` sequences. You can use the `--debug` flag to save visual representations of the pipeline.
```bash
python dataset/preprocess_jester.py \
    --input_dir /path/to/archive/Train \
    --csv_path /path/to/archive/Train.csv \
    --output_dir data/processed \
    --seq_len 30 \
    --stride 5 \
    --workers 4
```

### 2. Dataset Repair (Post-Hoc)
Run the dataset fixer to apply temporal carry-forward on dropped frames and prepare `X_fixed.npy`.
```bash
python fix_dataset.py
```

### 3. Training
Run the optimized training script (which automatically subsets to the 5 target classes and computes advanced features):
```bash
python train.py
```
This will:
- Train the BiLSTM model.
- Save the best weights to `weights/best_lstm_model.pth`.
- Generate a validation classification report and save the confusion matrix to `confusion_matrix.png`.

## Architecture Overview

- `dataset/keypoint_pipeline.py`: Extracts shoulder, wrist, and hand keypoints.
- `dataset/feature_builder.py`: Computes base relative coordinates and stabilizes frames.
- `train.py`: Contains the `GestureDataset` class (which computes velocities, arm extensions, and augmentations), the `GestureLSTM` model definition, and the main PyTorch orchestration loop.
