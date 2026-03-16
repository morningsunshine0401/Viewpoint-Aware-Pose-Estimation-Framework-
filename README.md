# Viewpoint-Aware Pose Estimation Framework

## Overview

Viewpoint-Aware Pose Estimation Framework is a real-time 6-DOF (6 Degrees of Freedom) pose estimation system designed specifically for aircraft tracking. It combines classical robotics techniques with modern deep learning to achieve robust, low-latency pose estimation with proper handling of vision processing delays.

### Key Features

- 🚁 **Real-time Aircraft Tracking**: Specialized for aircraft pose estimation with 14 viewpoint-specific anchors
- ⏱️ **Timestamp-Aware Processing**: Canonical VIO/SLAM approach for handling vision latency
- 🧠 **Enhanced Unscented Kalman Filter**: Variable-dt prediction with fixed-lag buffer for out-of-sequence measurements
- 🎯 **Multi-threaded Architecture**: Optimized for both low-latency display (30 FPS) and accurate processing
- 🔧 **Physics-Based Filtering**: Rate limiting prevents impossible orientation/position jumps
- 📊 **Adaptive Viewpoint Selection**: Intelligent switching between 14 pre-computed viewing angles

---

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   MainThread    │    │ ProcessingThread │
│   (30 FPS)      │    │   (Variable)     │
│                 │    │                  │
│ • Camera capture│    │ • YOLO detection │
│ • Timestamp     │ ┌──│ • Feature match  │
│ • Visualization │ │  │ • Pose estimation│
│ • UKF prediction│ │  │ • UKF update     │
└─────────────────┘ │  └─────────────────┘
         │           │           │
         └─── Queues + Locks ────┘
                   │
            ┌─────────────┐
            │   Enhanced  │
            │     UKF     │
            │(Timestamp-  │
            │   Aware)    │
            └─────────────┘
```

### Multi-Threading Design

- **MainThread**: High-frequency capture and display (30 FPS) with immediate timestamp recording
- **ProcessingThread**: AI-heavy computation (YOLO + SuperPoint + LightGlue + PnP) with timestamp-aware updates
- **Enhanced UKF**: Handles measurements at correct historical times with variable-dt motion models

---

## Technical Innovation

### Timestamp-Aware Vision Processing

Unlike traditional pose estimation systems that suffer from vision latency, VAPE MK53 implements the canonical VIO/SLAM approach:

1. **Immediate Timestamp Capture**: `t_capture = time.monotonic()` recorded the moment frames are obtained
2. **Latency-Corrected Updates**: UKF processes measurements at their actual capture time, not processing time
3. **Fixed-Lag Buffer**: 200-frame history enables handling of out-of-sequence measurements
4. **Variable-dt Motion Model**: Adapts to actual time intervals instead of assuming fixed frame rates

### Enhanced Unscented Kalman Filter

**State Vector (16D):**
```python
# [0:3]   - Position (x, y, z)
# [3:6]   - Velocity (vx, vy, vz)  
# [6:9]   - Acceleration (ax, ay, az)
# [9:13]  - Quaternion (qx, qy, qz, qw)
# [13:16] - Angular velocity (wx, wy, wz)
```

**Key Features:**
- **dt-Scaled Process Noise**: `Q_scaled = Q * dt + Q * (dt²) * 0.5`
- **Quaternion Normalization**: Prevents numerical drift
- **Rate Limiting**: Physics-based constraints prevent impossible motions
- **Robust Covariance**: SVD fallback for numerical stability

---

## Computer Vision Pipeline

### 1. Multi-Scale Object Detection
- **YOLO v8**: Custom trained on aircraft ("iha" class)
- **Adaptive Thresholding**: 0.30 → 0.20 → 0.10 confidence cascade
- **Largest-Box Selection**: Focuses on primary aircraft target

### 2. Deep Feature Extraction & Matching
- **SuperPoint**: CNN-based keypoint detector (up to 2048 keypoints)
- **LightGlue**: Attention-based feature matching with early termination
- **14 Viewpoint Anchors**: Pre-computed reference images for different viewing angles

### 3. Robust Pose Estimation
- **EPnP + RANSAC**: Initial pose estimation with outlier rejection
- **VVS Refinement**: Virtual Visual Servoing for sub-pixel accuracy
- **Temporal Consistency**: Viewpoint selection with failure recovery

### 4. Intelligent Viewpoint Management
```python
viewpoints = ['NE', 'NW', 'SE', 'SW', 'E', 'W', 'N', 'S', 
              'NE2', 'NW2', 'SE2', 'SW2', 'SU', 'NU']
```
- **Temporal Consistency**: Stick with working viewpoint
- **Adaptive Search**: Switch when current viewpoint fails
- **Quality Metrics**: Match count, inlier count, reprojection error

---

## Installation

### Requirements

**Python Version**: 3.11+

**Hardware Requirements**:
- NVIDIA GPU with CUDA 12.2+ (recommended)
- 8GB+ RAM
- USB camera or video input

### Dependencies

```bash
# Core Dependencies
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Computer Vision & AI
pip install ultralytics>=8.0.0
pip install lightglue
pip install opencv-python>=4.8.0

# Scientific Computing
pip install numpy>=1.24.0
pip install scipy>=1.11.0

# Utilities
pip install matplotlib>=3.7.0
```

### Required Files

1. **YOLO Model**: `best.pt` (trained aircraft detection model)
2. **Anchor Images**: 14 viewpoint reference images (NE.png, NW.png, etc.)
3. **Input Video**: Your aircraft footage for processing

---
