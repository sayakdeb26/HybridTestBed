# Implementation Proposal for Phase 1 Preparations

## 1. Pipeline Timing Instrumentation (Latency Tracking)
**Objective:** Capture T0 (Dataset Input) through T6 (Intent Logged) to compute LSTM, VLM, Hybrid, and Escalated Hybrid latency.

**Proposed Changes:**
1. Upgrade messages: Create a `amr_interfaces/msg/Timing` or inject `stamp` fields at every stage.
2. `dataset_source_node`: Tag `sensor_msgs/Image` with `T0 = now()`.
3. `keypoint_extractor_node`: Propagate T0. Add T1 after extraction. 
4. `sequence_buffer_node`: Propagate T0, T1. Add T2 after sequence array is formed.
5. `lstm_inference_node`: Add T3. Compute `LSTM latency = T3 - T0`.
6. `bridge_node`: Add T4 when VLM trigger is generated. Add T5 when VLM response is received. Add T6 when Intent is published to `/intents_raw`.
7. Extract values in the `evaluation_node` or `central_db_node` and write to `experiment_results/latency/latency.csv`.

## 2. Resource Monitoring
**Objective:** Track CPU/RAM and GPU/VRAM/Power/Temp. Export to `resource_usage.csv`.

**Proposed Changes:**
1. Create a new ROS2 node or a background bash script `resource_monitor_node.py`.
2. Use `psutil` for CPU/RAM utilization.
3. Use `pynvml` (NVIDIA Management Library) for GPU stats.
4. Log metrics every 1.0 second to `experiment_results/resource_usage/resource_usage.csv`.

## 3. Confidence Logging
**Objective:** For every prediction, store video_id, true label, predicted label, confidence, correctness.

**Proposed Changes:**
1. The `evaluation_node` or `central_db_node` will maintain `experiment_results/confidence/confidence_logs.csv`.
2. We will append the inference results directly there. `true label` must be fetched from the dataset manifest based on `video_id` (or `window_id` / `session_id`).

## 4. Hybrid Escalation Logging
**Objective:** Record details when LSTM escalates to VLM.

**Proposed Changes:**
1. In `bridge_node.py` and `lstm_inference_node.py`, capture the initial LSTM prediction and confidence before escalating.
2. Publish these fields in the `Intent` or a dedicated `EscalationLog` message.
3. Log to `experiment_results/escalation/hybrid_escalation_log.csv`. 
4. Calculate aggregate metrics (escalation count, % corrected, etc.) using a Python summary script post-experiment.

## 5. Confusion Matrices
**Objective:** Automatic generation of PNG and CSV for each experiment run (LT1-3, VT1-3, HT1-3).

**Proposed Changes:**
1. Create an `evaluator.py` script that takes the `confidence_logs.csv` or `hybrid_escalation_log.csv`.
2. It uses `sklearn.metrics.confusion_matrix` and `seaborn` to generate a heatmap and saves it into `experiment_results/confusion_matrices/`.
3. We will tie this into the `on_task_end()` lifecycle hook mentioned in the Continual Learning plan.

## Next Steps
Awaiting your approval to implement these modifications directly into the ROS2 nodes (`lstm_node.py`, `bridge_node.py`, `central_db_node.py`, etc.) and to push the resource monitor node.
