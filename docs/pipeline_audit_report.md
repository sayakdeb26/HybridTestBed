# Pipeline Audit Report
## Architecture Overview
The current workspace operates a 10-node hybrid pipeline combining lightweight LSTM inference with a heavy FastVLM path for uncertain gestures.

### Node Graph
- **dataset_source_node**: Sources images -> `/camera/image_raw`
- **keypoint_extractor_node**: Extracts 144-dim hand/pose keypoints -> `/keypoints`
- **sequence_buffer_node**: Aggregates sliding window (30 frames) of 296-dim features -> `/sequence`
- **lstm_inference_node**: Predicts using PyTorch. If confidence > 0.60, sends to `/gesture/prediction` & `/intents_raw`. Else, sends to `/lstm/unknown` to trigger VLM.
- **bridge_node**: Orchestrates VLM fallback. Requests video via `/recorder/request`, calls `/vlm/infer` service, and issues UI confirmations.
- **recorder_node**: Buffers frames. On request, encodes 4-second MP4 clip.
- **vlm_node**: FastVLM-1.5B node running on Apple MLX in a separate virtual env.
- **ui_kiosk_node**: Web server for HITL confirmation.
- **evaluation_node**: Sinks predictions to terminal.
- **central_db_node**: Stores intents and training examples in SQLite (`amr_db/amr_gestures.db`).

### What Works
- Main inference path (Sequence -> LSTM -> Prediction).
- VLM Escalation logic with Human-in-the-Loop Confirmation.
- Database logging for `Intent` and `TrainingExample`.
- The `bridge_node` robustly handles timeouts and system state.

### What Needs Modification for Phase 1
- **Instrumentation**: The pipeline does not currently propagate timestamps (T0 to T6) for latency computation across the system. We will need to upgrade `Float32MultiArray` to custom messages or stamp existing ones.
- **Logging Infrastructure**: Output needs to be redirected into the new `/experiment_results` directory.
- **Continual Learning**: Training loops currently exist in `hand_gesture_lab`, but online EWC and replay buffer logic must be integrated seamlessly.
