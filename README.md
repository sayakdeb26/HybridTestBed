# HyRes: Real-Time Hand Gesture & ROS 2 Pipeline

This repository contains a complete end-to-end framework for real-time hand gesture recognition, continual learning integration, and its integration into a ROS 2 autonomous robotics pipeline.

## 📂 Repository Structure

### 1. [hand_gesture_lab/](./hand_gesture_lab/)
The research, training, and continual learning environment for the gesture classification model.
- **Goal**: Trains a high-accuracy gesture classifier with continual learning capability.
- **Continual Learning**: Implements Elastic Weight Consolidation (EWC) and Replay Buffer methodologies to prevent catastrophic forgetting.
- **Key Scripts**:
  - `train.py`: Main training orchestration with advanced feature engineering.
  - `train_continual.py`: Orchestrates continual learning training tasks (e.g. L-T1 -> L-T2 -> L-T3) and handles EWC state chaining.
  - `fix_dataset.py`: Temporal carry-forward imputation for missing keypoints.
  - `ab_test.py`: Fast diagnostic training and feature comparison.

### 2. [gesture_ws/](./gesture_ws/)
The ROS 2 Humble workspace for live robot control and inference.
- **Packages**:
  - `keypoint_extractor_pkg`: Uses MediaPipe to extract coordinates from `/image`.
  - `sequence_buffer_pkg`: Maintains a rolling 30-frame window of features.
  - `lstm_inference_pkg`: Performs real-time classification, publishes `/prediction` and `/lstm/unknown` on low-confidence triggers.
  - `data_source_pkg`: Handles real-time image acquisition and injects `T0` timestamp.
  - `vlm_bridge_pkg`: Coordinates clip recording, calls the VLM service, and acts as the HITI (Human-in-the-Loop) state machine.
  - `amr_interfaces`: Declares custom message types (`InstrumentedKeypoints`, `InstrumentedSequence`) for high-precision latency instrumentation.

### 3. Extra Utilities & Monitoring
- `resource_monitor.py`: Background monitoring script tracking CPU, RAM, and GPU stats (VRAM, Power, Temperature) at 1s intervals.
- `generate_confusion_matrices.py`: Auto-generates confusion matrices (CSV and PNG) for pure LSTM and Hybrid decisions.

---

## ⚡ Continual Learning Strategy & State Persistence
To support sequential training across tasks (e.g. retraining on new gestures) without forgetting old ones, the pipeline implements a **Mixed Strategy**:
- **EWC (Elastic Weight Consolidation)**: Calculates and registers the Fisher Information Matrix and optimal parameters ($\theta^*$) for old tasks to penalize changes to important weights.
- **Replay Buffer**: Employs class-balanced prioritization for active gestures.
- **Persistence**: All state is preserved across training executions inside `hand_gesture_lab/weights/cl_state.pth`.

---

## ⏱️ Custom Timing Instrumentation (T0-T6)
To measure latency with nanosecond precision without Float32 conversion errors, the pipeline propagates timing parameters across ROS 2 nodes using custom `int64` timestamp fields:
- `T0` (Dataset Input): Recorded inside the Camera Image Header stamp.
- `T1` (Keypoint Extraction): Set in `keypoint_extractor_node`.
- `T2` (Sequence Construction): Set in `sequence_buffer_node`.
- `T3` (LSTM Prediction): Set in `lstm_inference_node`.
- `T4` (VLM Escalation Trigger): Recorded in `bridge_node` right before the asynchronous VLM call.
- `T5` (VLM Response): Recorded in `bridge_node` when the VLM client receives the result.
- `T6` (Intent Logged): Recorded when the user action is logged (either post-confirmation or directly from LSTM).

---

## 📊 Logging & Outputs
All experimental data is written to `/home/sayak/HyRes/experiment_results/`:
- **Confidence logs**: Saved to `experiment_results/confidence/confidence_logs.csv`.
- **Escalation logs**: Saved to `experiment_results/escalation/hybrid_escalation_log.csv`.
- **System Resource Metrics**: Saved to `experiment_results/resource_usage/resource_log.csv`.
- **Confusion Matrices**: Generated dynamically into `experiment_results/confusion_matrices/`.

---

## 🔍 Pre-Experiment Validation & Gaps Resolved
Before running the Phase 1 experiments, a validation pass resolved the following:
- **`train_continual.py` Role**: Documented as currently operating in validation mode (with synthetic data) to test strategy functionality. For Prompt 2 production runs, it will be modified to ingest actual `X.npy`/`y.npy` partitions defined in `dataset_manifest.csv`.
- **VLM Callback Crash**: Resolved a critical `NameError` inside `bridge_node.py`'s `vlm_response_callback` by calling `resp = future.result()` before checking attributes.
- **Checkpoint Versioning**: Implemented dynamic versioned paths to avoid overwriting CL state. Checkpoints are automatically saved as `best_lstm_model_rt{1,2,3}.pth` and `cl_state_rt{1,2,3}.pth`.
- **Pre-Escalation Logging**: Configured `lstm_inference_node.py` to forward the actual top-1 LSTM predicted label and confidence instead of `"UNCERTAIN"`. This enables comparison of pre- and post-escalation choices.
- **Threshold Parameterization**: Hardcoded LSTM confidence threshold (`0.60`) converted to a ROS 2 parameter (`confidence_threshold`) allowing custom runtime tuning.

---

## 🚀 Phase 1 Continual Learning Execution Complete
The **Phase 1 LSTM Continual Learning experiment** has been successfully executed, with full training and evaluation on 18,752 samples across four dynamic splits. 
- **Model Evolution**: Trained a sequence of models (`FUSE0` -> `FUSE33` -> `FUSE66` -> `FUSE100`).
- **Minimal Forgetting**: The integration of EWC and Replay Buffer proved successful; `FUSE100` exhibited only a `-1.08%` accuracy drift compared to the `FUSE0` baseline, explicitly mitigating catastrophic forgetting.
- **Results & Deliverables**: All metrics, latency logs, confusion matrices, prediction records, and analysis reports are published and categorized inside `/experiment_results/`:
  - `predictions/`: Raw sequence predictions and confidences.
  - `confusion_matrices/`: PNG heatmaps and CSV matrices.
  - `reports/training/`: Training and baseline generation logs.
  - `reports/evaluation/`: Testing and performance statistics per task.
  - `reports/analysis/`: Forgetting analysis, summary metrics, and CL data.

---

## 🤖 Phase 2: VLM Migration, Temporal Sampling & Production Replay Benchmarks

To address LSTM prediction uncertainty under live conditions, a hybrid decision loop was developed to escalate low-confidence sequences to Vision-Language Models (VLMs). Multiple advanced VLM architectures were integrated and benchmarking pipelines established:

### 1. VLM Models Integrated
*   **Qwen3-VL (4B & 8B)**: Implements closed-set temporal reasoning over frame sequences, verified using robust preflight validation paths.
*   **Video-LLaMA 3 (7B)**: Implements specialized video-language comprehension, demonstrating strong zero-shot classification on gesture actions.
*   **InternVideo2-Chat-8B**: Leverages a hybrid encoder-decoder architecture. Optimizations include:
    *   **PyTorch Native SDPA Patching**: Replaces naive attention layers with Scaled Dot-Product Attention for optimal memory efficiency.
    *   **Offloading & Memory Fallback Policy**: Auto-configured memory routing with dynamic GPU priority allocation and CPU fallback (60/40 split) to handle high VRAM consumption.
    *   **DynamicCache Compatibility**: Bypasses Cache size constraints to ensure seamless generation.

### 2. Temporal Frame Sampling Strategy Comparison
Evaluated gesture recognition performance using two frame extraction strategies across validation subsets:
*   **Uniform Sampling (`UNIFORM_20`)**: Selects 20 evenly distributed frames.
*   **Median-Centered Window (`MEDIAN_WINDOW_21`)**: Focuses on a 21-frame window centered on the peak of the gesture action.
*   *Verdict*: The Median-Centered Window significantly improves VLM classification accuracy, precision, and recall by isolating the high-action motion segments.

### 3. Production Replay Mode Benchmark (`PRODUCTION_REPLAY_MODE`)
Simulates live ROS 2 system conditions by executing:
1.  **Window Extraction**: Extracts a 5-frame event window from a rolling buffer around a detected gesture.
2.  **MP4 Generation**: Transcodes frames into standard MP4 format dynamically.
3.  **Cross-Model Benchmarking**: Feeds clips to FastVLM, Qwen3-VL, and Video-LLaMA3 to compare classification accuracy, latency, and resource footprint.

All Phase 2 reports are categorized inside `experiment_results/`:
*   `experiment_results/videollama3_smoke/`: Classification metrics, resources, latency, and video ingestion logs for Video-LLaMA 3.
*   `experiment_results/sampling_comparison/`: Comparison reports and prediction datasets between Uniform and Median-Centered sampling.
*   `experiment_results/replay_mode/`: Production replay metrics, confusion matrices, and the exported MP4 video clips.

---

## 🛠️ Getting Started

### Training & Continual Retraining
1. Navigate to `hand_gesture_lab/`.
2. Ensure data is prepared in `data/processed/`.
3. Run `python3 train_continual.py` to initiate tasks retraining. EWC state and replay buffers will automatically load and save to versioned checkpoints (`weights/cl_state_rt*.pth`).

### Launching the ROS 2 Pipeline with Resource Monitor
1. Source ROS 2 Humble environment: `source /opt/ros/humble/setup.bash`.
2. Build the workspace: `colcon build`.
3. Source the overlay: `source gesture_ws/install/setup.bash`.
4. Run `./run_ros2_test.sh` to spin up all nodes, which also starts the background resource monitor automatically.
5. To plot final metrics after stopping the pipeline, run `python3 generate_confusion_matrices.py`.

### VLM Preflight, Smoke Tests & Benchmarks
To execute Phase 2 validation and benchmarking runs:
1.  **Run Qwen3-VL Preflight & Smoke Test**:
    ```bash
    python3 run_qwen3_vl_preflight.py
    python3 run_qwen3_vl_smoke_test.py
    ```
2.  **Run Video-LLaMA 3 Smoke Test**:
    ```bash
    python3 run_videollama3_smoke_test.py
    ```
3.  **Run InternVideo2-Chat-8B Smoke Test**:
    ```bash
    python3 run_internvideo2_smoke_test.py
    ```
4.  **Run Temporal Sampling Strategy Comparison**:
    ```bash
    python3 run_sampling_comparison.py
    ```
5.  **Run Production Replay Mode Benchmark**:
    ```bash
    python3 run_replay_mode_benchmark.py
    ```

---
**Maintained by Sayak Deb**  
*Project Focused on: Continual Learning, Gesture-Based Robot Control, and Temporal Motion Analysis.*