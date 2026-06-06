# Continual Learning Integration Plan

## Executive Summary
We successfully located `mixed_strategy.py` inside the workspace root. Since the dependencies (`base_strategy.py`, `ewc_strategy.py`, and `replay_strategy.py`) were not present, we implemented them based on standard Continual Learning paradigms (Online EWC and Experience Replay) matching the expected API. We then verified the setup by creating and executing `hand_gesture_lab/train_continual.py` using the `MixedStrategy` class.

---

## 1. Task Definition and Transitions
- **How task boundaries should be defined**: A task boundary is defined at the end of each training phase over a new data block. During real-time deployment, low-confidence samples are escalated to FastVLM/UI and saved as validated instances. Once a sufficient block of new samples is accumulated (representing a new task), retraining is triggered.
- **How DS1, DS2, DS3 map to tasks**: 
  - **Task 0 (Baseline)**: Initial training on the standard `Train` dataset.
  - **Task 1**: Fine-tuning using `DS1` (collected during the first deployment stage).
  - **Task 2**: Fine-tuning using `DS2` (second stage).
  - **Task 3**: Fine-tuning using `DS3` (third stage).
- **How `on_task_end()` should be triggered**: The training script `train_continual.py` invokes `strategy.on_task_end(X_task, y_task)` at the end of the retraining epochs for the current task. This calculates the Fisher Information Matrix (FIM) and registers the task parameters.

---

## 2. Checkpoint Chaining & Replay Memory
- **Checkpoint Chaining**: In `train_continual.py`, versioned checkpoints are saved (e.g., `best_lstm_model_continual.pth`). Checkpoint chaining exists by loading the weights of the previous task model (`model_path`) before executing the next continual strategy stage.
- **Replay Memory Persistence**: The `ReplayBuffer` maintains a running cache of past training examples. To preserve this across separate training session runs, we can serialize `strategy.replay_buffer` to a file (e.g., `replay_buffer.pkl`) using Python's `pickle` library, allowing it to load back up seamlessly.
- **EWC Consolidation**: Online EWC consolidates parameter importance by keeping a running average of the Fisher Information Matrix across all tasks, regulated by `online_alpha` (set to `0.5`). During training, an additional quadratic penalty term is added to the standard Cross-Entropy loss function based on the parameters' deviations from their previous optimal states.

---

## 3. Mappings Verification
- **Class ID Mappings**:
  - `0`: Swiping Left (orig. 16)
  - `1`: Swiping Right (orig. 17)
  - `2`: Rolling Hand Forward (orig. 8)
  - `3`: Rolling Hand Backward (orig. 7)
  - `4`: Thumb Down (orig. 19)
  - `5`: Stop Sign (orig. 14)
- **Weak Class IDs**: Set to `(3, 4, 5)` in `mixed_strategy.py`. These map to:
  - `3`: Rolling Hand Backward
  - `4`: Thumb Down
  - `5`: Stop Sign
- **Validity**: These mappings align perfectly with `hand_gesture_lab/train.py`. The `ReplayBuffer` balanced sampling prioritizes keeping a higher portion of these weak classes to mitigate catastrophic forgetting of complex motion patterns.
