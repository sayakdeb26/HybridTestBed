# Timestamp Architecture Review

This report reviews and compares two options for implementing the timestamp instrumentation pipeline to capture T0 through T6 latency metrics.

---

## 1. Option Comparison

| Metric | Option A: Custom ROS2 Message | Option B: Appending to Float32MultiArray |
| :--- | :--- | :--- |
| **Maintainability** | **High**. Fields are explicitly defined with names (`t0_ns`, `t1_ns`, etc.). Easy to modify. | **Low**. Relies on hardcoded index offsets at the end of the data array. |
| **Debugging** | **Easy**. Can run `ros2 topic echo /topic` and see exactly which timestamps are set. | **Difficult**. Outputs a raw array of floats. Must search the tail of the array and cast floats back to ints. |
| **Serialization** | **Type-Safe**. Nanoseconds are stored as `int64`. No precision loss. | **Lossy**. Float32 has only 24 bits of mantissa (approx. 7 decimal digits). Storing epoch nanoseconds (19 digits) will cause severe truncation. |
| **Future Extensibility** | **High**. Easily add metadata, camera IDs, or additional timing stages. | **Low**. Modifying array structure breaks downstream node shape validation. |
| **Latency Overhead** | **Negligible**. Fast C++ and Python serialization built into ROS2 interfaces. | **Negligible**. Directly parses raw float buffer. |
| **ROS2 Best Practices** | **Aligned**. Encourages explicit structured messages rather than generic arrays. | **Violated**. Anti-pattern to mix metadata inside numeric sensor arrays. |

---

## 2. Recommendation & Justification

### Recommendation: **Option A (Custom ROS2 Message)**

### Justification:
The primary blocker for Option B is **numerical precision loss**. A standard Unix timestamp in nanoseconds (e.g., `1716298374000000000`) requires 64-bit integer representation. If stored as a 32-bit float (`Float32MultiArray`), it will be rounded to the nearest power of 2, losing up to several seconds of accuracy, making sub-millisecond latency tracking mathematically impossible. Option A provides structured, self-documenting data fields, prevents index-out-of-bounds errors, and enables precise int64 timekeeping.

---

## 3. Draft Message Definition

We recommend declaring a custom message in the `amr_interfaces` package:

### File: `amr_interfaces/msg/InstrumentedSequence.msg`
```msg
# Header containing standard ROS 2 stamp
std_msgs/Header header

# Flattened sequence data (30 frames x 296 dimensions)
float32[] sequence_data

# Timestamps in nanoseconds (Epoch time)
int64 t0_dataset_input
int64 t1_keypoint_extraction
int64 t2_sequence_construction
int64 t3_lstm_prediction
int64 t4_vlm_escalation_trigger
int64 t5_vlm_response
int64 t6_intent_logged
```
This single message can be passed through the pipeline, accumulating timestamps at each node before the final evaluation node logs the latencies.
