# ==================================================================================================
#
#  VAPE - REAL-TIME 6-DOF POSE ESTIMATOR (Enhanced with Timestamp Support)
#
#  Author: [Kim Young Run]
#  Description: This script performs real-time 6-DOF (position and orientation) pose estimation
#               of an aircraft using a multi-threaded architecture with proper timestamp handling
#               for vision latency correction following canonical VIO/SLAM approaches.
#
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
#  1. IMPORTS AND INITIAL SETUP
# --------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import torch
import time
import argparse
import warnings
import json
import threading
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import queue
import math
from collections import deque

# --- Dependency Imports ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 VAPE Pose Estimator (Enhanced with Timestamp Support)")
try:
    from ultralytics import YOLO
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    from scipy.spatial import cKDTree
    print("✅ All libraries loaded successfully.")
except ImportError as e:
    print(f"❌ Import error: {e}. Please run 'pip install -r requirements.txt' to install dependencies.")
    exit(1)


# --------------------------------------------------------------------------------------------------
#  2. UTILITY FUNCTIONS
# --------------------------------------------------------------------------------------------------
def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalizes a quaternion to ensure it has a unit length of 1.
    Assumes [x, y, z, w] order throughout the system.
    """
    norm = np.linalg.norm(q)
    if norm > 1e-10:
        return q / norm
    else:
        # Return identity quaternion [0, 0, 0, 1] in [x,y,z,w] order
        return np.array([0.0, 0.0, 0.0, 1.0])

def quaternion_to_xyzw(q_input):
    """
    Convert quaternion to [x,y,z,w] order if needed.
    Handles both [w,x,y,z] and [x,y,z,w] input orders.
    """
    q = np.array(q_input)
    if len(q) != 4:
        raise ValueError("Quaternion must have 4 components")
    
    # Check if this looks like [w,x,y,z] order (w component is typically largest for small rotations)
    if abs(q[0]) > abs(q[3]) and abs(q[0]) > 0.7:  # Heuristic: w component usually > 0.7 for small rotations
        # Convert [w,x,y,z] to [x,y,z,w]
        return np.array([q[1], q[2], q[3], q[0]])
    else:
        # Assume already in [x,y,z,w] order
        return q

def quat_mul(a, b):
    """Multiply two quaternions"""
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_conj(q):
    """Quaternion conjugate"""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_inv(q):
    """Quaternion inverse"""
    qn = normalize_quaternion(q)
    return quat_conj(qn)

def quat_to_axis_angle(q):
    """Convert quaternion to axis-angle representation"""
    qn = normalize_quaternion(q)
    w = float(np.clip(qn[3], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w*w, 0.0))
    axis = np.array([1.0, 0.0, 0.0]) if s < 1e-8 else qn[:3]/s
    return axis, angle

def axis_angle_to_quat(axis, angle):
    """Convert axis-angle to quaternion"""
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(angle/2.0)
    return normalize_quaternion(np.array([ax[0]*s, ax[1]*s, ax[2]*s, np.cos(angle/2.0)]))

def clamp_quaternion_towards(q_from, q_to, max_deg_per_s, dt):
    """Rate-limit quaternion rotation to prevent sudden orientation jumps."""
    # Keep quaternions in same hemisphere to avoid 180° sign jump
    if np.dot(q_from, q_to) < 0.0:
        q_to = -q_to
    
    # Calculate delta quaternion to go from predicted -> measured
    dq = quat_mul(quat_inv(q_from), q_to)
    axis, ang = quat_to_axis_angle(dq)  # ang in [0, pi]
    
    # Calculate maximum allowed angle for this time step
    ang_limit = np.deg2rad(max_deg_per_s) * max(dt, 1e-6)
    
    if ang > ang_limit:
        # Clamp the rotation to the maximum allowed
        dq = axis_angle_to_quat(axis, ang_limit)
        q_out = quat_mul(q_from, dq)
    else:
        q_out = q_to
    
    return normalize_quaternion(q_out)

def clamp_position_towards(pos_from, pos_to, max_speed_m_per_s, dt):
    """Rate-limit position movement to prevent sudden position jumps."""
    # Calculate the movement vector
    delta_pos = pos_to - pos_from
    distance = np.linalg.norm(delta_pos)
    
    # Calculate maximum allowed movement for this time step
    max_distance = max_speed_m_per_s * max(dt, 1e-6)
    
    if distance > max_distance:
        # Clamp the movement to the maximum allowed
        direction = delta_pos / distance  # Unit vector
        pos_out = pos_from + direction * max_distance
    else:
        pos_out = pos_to
    
    return pos_out


# --------------------------------------------------------------------------------------------------
#  3. DATA STRUCTURES
# --------------------------------------------------------------------------------------------------
@dataclass
class ProcessingResult:
    """Holds all data for a single processed frame for visualization and logging."""
    frame_id: int
    frame: np.ndarray
    position: Optional[np.ndarray] = None
    quaternion: Optional[np.ndarray] = None
    kf_position: Optional[np.ndarray] = None
    kf_quaternion: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    num_inliers: int = 0
    pose_success: bool = False
    viewpoint_used: Optional[str] = None
    capture_time: Optional[float] = None

@dataclass
class PoseData:
    """A simple container for the results of a successful pose estimation."""
    position: np.ndarray
    quaternion: np.ndarray
    inliers: int
    reprojection_error: float
    viewpoint: str
    total_matches: int
    coverage_score: float = 0.0


# --------------------------------------------------------------------------------------------------
#  4. ENHANCED UNSCENTED KALMAN FILTER WITH TIMESTAMP SUPPORT
# --------------------------------------------------------------------------------------------------
class UnscentedKalmanFilter:
    """
    Enhanced UKF with timestamp support for handling vision latency
    Following the canonical VIO/SLAM approach with fixed-lag buffer
    """
    def __init__(self, dt=1/15.0):
        self.dt = dt  # Default dt for display predictions
        self.initialized = False
        
        # Rate limiting parameters
        self.max_rot_rate_dps = 30.0
        self.max_pos_speed_mps = 1.5
        
        # State vector [pos(3), vel(3), acc(3), quat(4), ang_vel(3)]
        # Quaternion stored in [x,y,z,w] order at indices 9,10,11,12
        self.n = 16  # State size
        self.m = 7   # Measurement size [pos(3), quat(4)]

        # State vector init — quaternion in [x,y,z,w] order
        self.x = np.zeros(self.n)
        self.x[12] = 1.0  # Quaternion w component in [x,y,z,w] order (index 12 = 9+3)

        # State covariance matrix
        self.P = np.eye(self.n) * 0.1
        self.P0 = self.P.copy()  # Store initial covariance

        # UKF parameters
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n

        # Weights for sigma points
        self.wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lambda_)))
        self.wc = self.wm.copy()
        self.wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.wc[0] = self.lambda_ / (self.n + self.lambda_) + (1.0 - self.alpha**2 + self.beta)

        # Noise matrices
        self.Q = np.eye(self.n) * 1e-1 #1e-2  # Process noise
        self.R = np.eye(self.m) * 1e-4  # Measurement noise

        # Timestamp and history management
        self.t_state = None  # Current state time
        self.history = deque(maxlen=200)  # (t, x.copy(), P.copy()) buffer

    def _push_history(self):
        """Store current state in history buffer before predict"""
        if self.t_state is not None:
            self.history.append((self.t_state, self.x.copy(), self.P.copy()))

    def _generate_sigma_points(self, x, P):
        """Generates sigma points for UKF with robust Cholesky decomposition"""
        sigmas = np.zeros((2 * self.n + 1, self.n))
        
        # Ensure P is symmetric and positive definite
        P_sym = 0.5 * (P + P.T)
        
        try:
            U = np.linalg.cholesky((self.n + self.lambda_) * P_sym)
        except np.linalg.LinAlgError:
            # If Cholesky fails, add small jitter to diagonal
            P_jitter = P_sym + 1e-9 * np.eye(self.n)
            try:
                U = np.linalg.cholesky((self.n + self.lambda_) * P_jitter)
            except np.linalg.LinAlgError:
                # Fallback: use SVD
                print("⚠️ Cholesky failed, using SVD fallback")
                U_svd, s, _ = np.linalg.svd(P_jitter)
                U = U_svd @ np.diag(np.sqrt(np.maximum(s, 1e-12))) @ U_svd.T
                U = U * np.sqrt(self.n + self.lambda_)
        
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i+1] = x + U[:, i]
            sigmas[self.n+i+1] = x - U[:, i]
        return sigmas

    def motion_model(self, x_in, dt):
        """Motion model with variable dt and proper quaternion handling"""
        x_out = np.zeros_like(x_in)

        pos, vel, acc = x_in[0:3], x_in[3:6], x_in[6:9]
        x_out[0:3] = pos + vel * dt + 0.5 * acc * dt**2  # Position update
        x_out[3:6] = vel + acc * dt                      # Velocity update
        x_out[6:9] = acc                                 # Acceleration is constant

        q, w = x_in[9:13], x_in[13:16]
        # Quaternion integration from angular velocity using proper [x,y,z,w] order
        # q = [qx, qy, qz, qw] where qw is at index 3 (12 in full state)
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        omega_mat = 0.5 * np.array([
            [ qw, -qz,  qy],   # dq_x
            [ qz,  qw, -qx],   # dq_y
            [-qy,  qx,  qw],   # dq_z
            [-qx, -qy, -qz]    # dq_w
        ])
        
        # Simplified integration: q_new = q + 0.5 * dt * Ω(ω) * q
        q_dot = omega_mat @ w
        q_new = q + dt * q_dot
        
        # Normalize quaternion after integration
        x_out[9:13] = normalize_quaternion(q_new)
        x_out[13:16] = w  # Angular velocity is constant

        return x_out

    def predict(self, dt):
        """UKF Prediction Step with variable dt and dt-scaled process noise"""
        if not self.initialized:
            return self.x[0:3], self.x[9:13]

        x_prev = self.x.copy()

        # Generate sigma points from current state
        sigmas = self._generate_sigma_points(self.x, self.P)

        # Propagate each sigma point through motion model with given dt
        sigmas_f = np.array([self.motion_model(s, dt) for s in sigmas])

        # Recalculate mean and covariance
        x_pred = np.sum(self.wm[:, np.newaxis] * sigmas_f, axis=0)
        
        x_pred[9:13] = normalize_quaternion(x_pred[9:13])
        
        # dt-scaled process noise Q
        Q_scaled = self.Q * dt + self.Q * (dt**2) * 0.5
        
        P_pred = Q_scaled.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_f[i] - x_pred
            P_pred += self.wc[i] * np.outer(y, y)

        P_pred = 0.5 * (P_pred + P_pred.T)

        # Rate limiting on prediction output
        x_pred[0:3] = clamp_position_towards(
            x_prev[0:3], x_pred[0:3], self.max_pos_speed_mps, dt
        )
        x_pred[9:13] = clamp_quaternion_towards(
            x_prev[9:13], x_pred[9:13], self.max_rot_rate_dps, dt
        )

        self.x = x_pred
        self.P = P_pred
        
        return self.x[0:3], self.x[9:13]

    def hx(self, x_in):
        """Measurement function mapping state to measurement space with normalized quaternion"""
        z = np.zeros(self.m)
        z[0:3] = x_in[0:3]  # Position
        z[3:7] = normalize_quaternion(x_in[9:13])
        return z

    def _measurement_update(self, z_pos, z_quat, R):
        """Internal measurement update function with robust covariance handling"""
        # Apply hemisphere consistency for quaternion
        if self.initialized and np.dot(self.x[9:13], z_quat) < 0.0:
            z_quat = -z_quat

        measurement = np.concatenate([z_pos, normalize_quaternion(z_quat)])

        # Generate sigma points from predicted state
        sigmas_f = self._generate_sigma_points(self.x, self.P)

        # Transform sigma points to measurement space
        sigmas_h = np.array([self.hx(s) for s in sigmas_f])

        # Calculate predicted measurement and covariance
        z_pred = np.sum(self.wm[:, np.newaxis] * sigmas_h, axis=0)
        # Normalize predicted quaternion
        z_pred[3:7] = normalize_quaternion(z_pred[3:7])
        
        S = R.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_h[i] - z_pred
            S += self.wc[i] * np.outer(y, y)

        # Ensure S is symmetric
        S = 0.5 * (S + S.T)

        # Calculate cross-covariance
        P_xz = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            y_x = sigmas_f[i] - self.x
            y_z = sigmas_h[i] - z_pred
            P_xz += self.wc[i] * np.outer(y_x, y_z)

        # Kalman update
        try:
            K = P_xz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if S is singular
            K = P_xz @ np.linalg.pinv(S)
        
        innovation = measurement - z_pred
        self.x += K @ innovation
        
        # Normalize quaternion after update
        self.x[9:13] = normalize_quaternion(self.x[9:13])
        
        # Update covariance
        self.P -= K @ S @ K.T
        
        # Ensure P stays symmetric positive-definite
        self.P = 0.5 * (self.P + self.P.T)
        
        # Add small jitter to diagonal if needed to maintain positive definiteness
        min_eigenval = np.min(np.real(np.linalg.eigvals(self.P)))
        if min_eigenval < 1e-12:
            self.P += (1e-9 - min_eigenval) * np.eye(self.n)

    def update_with_timestamp(self, z_pos, z_quat, t_meas, R=None, t_now=None):
        """
        Time-aware update following canonical VIO/SLAM approach.
        Rate limiting is applied in predict(), not here.
        """
        if R is None:
            R = self.R.copy()

        # First valid measurement — initialize filter
        if self.t_state is None:
            self.x[0:3] = z_pos
            self.x[9:13] = normalize_quaternion(z_quat)
            self.P[:] = self.P0
            self.t_state = t_meas
            self.initialized = True
            return self.x[0:3], self.x[9:13]

        # Case A: Measurement is in the future or present
        if t_meas >= self.t_state:
            dt1 = t_meas - self.t_state
            if dt1 > 0:
                self._push_history()
                self.predict(dt1)
                self.t_state = t_meas

            self._measurement_update(z_pos, normalize_quaternion(z_quat), R)

            # Fast-forward to current time for display
            if t_now is not None and t_now > t_meas:
                dt2 = t_now - t_meas
                self._push_history()
                self.predict(dt2)
                self.t_state = t_now

            return self.x[0:3], self.x[9:13]

        # Case B: Measurement is in the past (out-of-sequence)
        valid_history = [(i, t, x, P) for i, (t, x, P) in enumerate(self.history) if t <= t_meas]
        
        if not valid_history:
            dt1 = max(0, t_meas - self.t_state)
            if dt1 > 0:
                self._push_history()
                self.predict(dt1)
                self.t_state = t_meas
            
            self._measurement_update(z_pos, normalize_quaternion(z_quat), R)
            return self.x[0:3], self.x[9:13]

        # Roll back to the historical state closest to t_meas
        k, t_k, x_k, P_k = valid_history[-1]
        self.x[:] = x_k
        self.P[:] = P_k
        self.t_state = t_k

        if t_meas > t_k:
            self.predict(t_meas - t_k)
            self.t_state = t_meas

        self._measurement_update(z_pos, normalize_quaternion(z_quat), R)

        # Replay forward to restore current time
        for j in range(k + 1, len(self.history)):
            t_next = self.history[j][0]
            dt_replay = t_next - self.t_state
            if dt_replay > 0:
                self.predict(dt_replay)
                self.t_state = t_next

        return self.x[0:3], self.x[9:13]

    def predict_to_time(self, t_target):
        """Predict state to a specific target time (for display) without modifying filter state"""
        if self.t_state is None or not self.initialized:
            return self.x[0:3], self.x[9:13]

        dt = t_target - self.t_state
        if dt <= 0:
            return self.x[0:3], self.x[9:13]

        # Temporarily predict forward without modifying the filter
        x_temp = self.x.copy()
        P_temp = self.P.copy()
        
        # Generate sigma points and predict
        sigmas = self._generate_sigma_points(x_temp, P_temp)
        sigmas_f = np.array([self.motion_model(s, dt) for s in sigmas])
        x_pred = np.sum(self.wm[:, np.newaxis] * sigmas_f, axis=0)
        
        # Normalize quaternion in prediction
        x_pred[9:13] = normalize_quaternion(x_pred[9:13])
        
        return x_pred[0:3], x_pred[9:13]

    # Legacy methods for backward compatibility
    def update(self, position: np.ndarray, quaternion: np.ndarray):
        """Legacy update method - uses current time"""
        t_now = time.time()
        return self.update_with_timestamp(position, quaternion, t_now, t_now=t_now)

    def set_rate_limits(self, max_rotation_dps: float = None, max_position_mps: float = None):
        """Adjust rate limits"""
        if max_rotation_dps is not None:
            self.max_rot_rate_dps = max_rotation_dps
            print(f"🔄 Rotation rate limit set to {max_rotation_dps}°/s")
            
        if max_position_mps is not None:
            self.max_pos_speed_mps = max_position_mps
            print(f"🎯 Position speed limit set to {max_position_mps} m/s")

    def set_rotation_rate_limit(self, max_degrees_per_second: float):
        """Legacy method for backward compatibility"""
        self.set_rate_limits(max_rotation_dps=max_degrees_per_second)


# --------------------------------------------------------------------------------------------------
#  5. HIGH-FREQUENCY MAIN THREAD (ENHANCED)
# --------------------------------------------------------------------------------------------------
class MainThread(threading.Thread):
    """
    Handles high-frequency, low-latency tasks with timestamp capture
    """
    def __init__(self, processing_queue, visualization_queue, pose_data_lock, kf, args):
        super().__init__()
        self.running = True
        self.processing_queue = processing_queue
        self.visualization_queue = visualization_queue
        self.pose_data_lock = pose_data_lock
        self.kf = kf
        self.args = args

        self.camera_width, self.camera_height = 1280, 720
        self.is_video_stream = False
        self.video_capture = None
        self.image_files = []
        self.frame_idx = 0
        self.frame_count = 0
        self.start_time = time.time()

        self._initialize_input_source()
        self.K, self.dist_coeffs = self._get_camera_intrinsics()

    def _initialize_input_source(self):
        """Initializes the input source based on command-line arguments."""
        if self.args.webcam:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened(): raise IOError("Cannot open webcam.")
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.is_video_stream = True
            print("📹 Using webcam input.")
        elif self.args.video_file:
            if not os.path.exists(self.args.video_file): raise FileNotFoundError(f"Video file not found: {self.args.video_file}")
            self.video_capture = cv2.VideoCapture(self.args.video_file)
            self.is_video_stream = True
            print(f"📹 Using video file input: {self.args.video_file}")
        elif self.args.image_dir:
            if not os.path.exists(self.args.image_dir): raise FileNotFoundError(f"Image directory not found: {self.args.image_dir}")
            self.image_files = sorted([os.path.join(self.args.image_dir, f) for f in os.listdir(self.args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not self.image_files: raise IOError(f"No images found in directory: {self.args.image_dir}")
            print(f"🖼️ Found {len(self.image_files)} images for processing.")
        else:
            raise ValueError("No input source specified. Use --webcam, --video_file, or --image_dir.")

    def _get_next_frame(self):
        """Fetches the next frame from the configured input source."""
        if self.is_video_stream:
            ret, frame = self.video_capture.read()
            return frame if ret else None
        else:
            if self.frame_idx < len(self.image_files):
                frame = cv2.imread(self.image_files[self.frame_idx])
                self.frame_idx += 1
                return frame
            return None

    def run(self):
        """Updated main loop with timestamp capture and time-aware prediction"""
        window_name = "VAPE - Real-time Pose Estimation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.camera_width, self.camera_height)

        while self.running:
            loop_start_time = time.time()

            # Get frame and capture timestamp immediately
            frame = self._get_next_frame()
            if frame is None: 
                break
            
            t_capture = time.monotonic()

            # 1. Kalman Filter Prediction to current time for display
            t_now = time.monotonic()  # Use monotonic time consistently
            with self.pose_data_lock:
                # Use predict_to_time for display without modifying filter state
                predicted_pose_tvec, predicted_pose_quat = self.kf.predict_to_time(t_now)

            # 2. Visualization
            vis_frame = frame.copy()
            if predicted_pose_tvec is not None and predicted_pose_quat is not None:
                self._draw_axes(vis_frame, predicted_pose_tvec, predicted_pose_quat)

            # Draw OSD info
            elapsed_time = time.time() - self.start_time
            fps = (self.frame_count + 1) / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(vis_frame, "STATUS: PREDICTING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            
            # Show filter state time info
            if self.kf.t_state is not None:
                age_ms = (t_now - self.kf.t_state) * 1000
                cv2.putText(vis_frame, f"Filter Age: {age_ms:.1f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            self.frame_count += 1

            # 3. Send frame WITH TIMESTAMP to processing thread
            if self.processing_queue.qsize() < 2:
                
                self.processing_queue.put((frame.copy(), t_capture))

            # 4. Handle visualization
            if self.args.show:
                try:
                    vis_data = self.visualization_queue.get_nowait()
                    kpts, vis_crop = vis_data['kpts'], vis_data['crop']
                    for kpt in kpts:
                        cv2.circle(vis_crop, (int(kpt[0]), int(kpt[1])), 2, (0, 255, 0), -1)
                    cv2.imshow("SuperPoint Features", vis_crop)
                except queue.Empty:
                    pass

            # 5. Display and exit check
            cv2.imshow(window_name, vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                print("User requested shutdown.")
                break

            # 6. Frame rate cap
            frame_rate_cap = 30.0
            time_to_wait = (1.0 / frame_rate_cap) - (time.time() - loop_start_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        self.cleanup()

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        """Returns the camera intrinsic matrix K and distortion coefficients."""
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([
             0.02692405, -0.03433880, 0.01104186, 0.00124234, -0.12498783
         ], dtype=np.float32)
        return K, dist_coeffs
    
    

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Converts a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
        q_norm = normalize_quaternion(q)
        x, y, z, w = q_norm
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

    def _draw_axes(self, frame: np.ndarray, position: np.ndarray, quaternion: np.ndarray):
        """Draws a 3D coordinate axis on the frame at the estimated pose."""
        try:
            R = self._quaternion_to_rotation_matrix(quaternion)
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            axis_pts = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.K, self.dist_coeffs)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            origin = tuple(img_pts[0])
            cv2.line(frame, origin, tuple(img_pts[1]), (0,0,255), 3)
            cv2.line(frame, origin, tuple(img_pts[2]), (0,255,0), 3)
            cv2.line(frame, origin, tuple(img_pts[3]), (255,0,0), 3)
        except (cv2.error, AttributeError, ValueError):
            pass

    def cleanup(self):
        """Releases resources upon shutdown."""
        self.running = False
        if self.is_video_stream and self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------------
#  6. LOW-FREQUENCY PROCESSING THREAD (ENHANCED)
# --------------------------------------------------------------------------------------------------
class ProcessingThread(threading.Thread):
    """
    Handles low-frequency, computationally-heavy tasks with timestamp-aware UKF updates
    """
    def __init__(self, processing_queue, visualization_queue, pose_data_lock, kf, args):
        super().__init__()
        self.running = True
        self.processing_queue = processing_queue
        self.visualization_queue = visualization_queue
        self.pose_data_lock = pose_data_lock
        self.kf = kf
        self.args = args

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.camera_width, self.camera_height = 1280, 720
        self.all_poses_log = []

        # Temporal Consistency for Viewpoint Selection (Algorithm 1)
        self.current_best_viewpoint = None
        self.needs_full_eval = True  # Start with full evaluation
        self.theta_reuse = 12.0     # θ_reuse: minimum score to allow viewpoint reuse

        # Pre-filtering for Measurement Rejection
        self.last_orientation: Optional[np.ndarray] = None
        self.ORI_MAX_DIFF_DEG = 30
        self.rejected_consecutive_frames_count = 0
        self.MAX_REJECTED_FRAMES = 5

        self.yolo_model = None
        self.extractor = None
        self.matcher = None
        self.viewpoint_anchors = {}

        self._initialize_models()
        self._initialize_anchor_data()
        self.K, self.dist_coeffs = self._get_camera_intrinsics()

    def _initialize_models(self):
        """Loads all required machine learning models onto the selected device."""
        print("📦 Loading models...")
        self.yolo_model = YOLO("best.pt").to(self.device)
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        print("   ...models loaded.")

    def _initialize_anchor_data(self):
        """Pre-processes anchor data for each viewpoint."""
        print("🛠️ Initializing anchor data...")
        
        # Hardcoded anchor data (same as before)
        ne_anchor_2d = np.array([[928, 148],[570, 111],[401, 31],[544, 141],[530, 134],[351, 228],[338, 220],[294, 244],[230, 541],[401, 469],[414, 481],[464, 451],[521, 510],[610, 454],[544, 400],[589, 373],[575, 361],[486, 561],[739, 385],[826, 305],[791, 285],[773, 271],[845, 233],[826, 226],[699, 308],[790, 375]], dtype=np.float32)
        ne_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.000, -0.000, 0.156],[-0.014, 0.000, 0.156],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.074],[-0.014, 0.000, 0.042],[0.000, 0.000, 0.042],[-0.080, -0.000, 0.156],[-0.100, -0.030, 0.000],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.042],[-0.090, -0.000, -0.042]], dtype=np.float32)
        nw_anchor_2d = np.array([[511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213], [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388], [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143], [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]], dtype=np.float32)
        nw_anchor_3d = np.array([[-0.014, 0.0, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.0, -0.042], [-0.014, 0.0, 0.156], [-0.023, 0.0, -0.065], [0.0, 0.0, -0.156], [0.025, 0.0, -0.015], [0.217, 0.0, 0.07], [0.23, 0.0, -0.07], [-0.014, 0.0, -0.156], [0.0, 0.0, 0.042], [-0.057, -0.018, -0.01], [-0.074, -0.0, 0.128], [0.206, -0.07, -0.002], [-0.0, -0.0, 0.156], [-0.017, -0.0, -0.092], [0.217, -0.0, -0.027], [-0.052, -0.0, -0.097], [-0.019, -0.0, 0.128], [-0.035, -0.018, -0.01], [0.217, -0.0, -0.07], [-0.08, -0.0, 0.156], [0.23, 0.0, 0.07], [-0.023, -0.0, -0.075], [-0.029, -0.0, -0.127], [-0.09, -0.0, -0.042], [0.206, -0.055, -0.002], [-0.09, -0.0, -0.015], [0.0, -0.0, -0.015], [-0.037, -0.0, -0.097], [-0.074, -0.0, 0.074], [-0.019, -0.0, 0.074], [0.23, -0.0, -0.113], [-0.1, -0.03, 0.0], [0.17, -0.0, -0.015], [0.23, -0.0, 0.113], [-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [-0.08, -0.0, -0.156]], dtype=np.float32)
        se_anchor_2d = np.array([[415, 144], [1169, 508], [275, 323], [214, 395], [554, 670], [253, 428], [280, 415], [355, 365], [494, 621], [519, 600], [806, 213], [973, 438], [986, 421], [768, 343], [785, 328], [841, 345], [931, 393], [891, 306], [980, 345], [651, 210], [625, 225], [588, 216], [511, 215], [526, 204], [665, 271]], dtype=np.float32)
        se_anchor_3d = np.array([[-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [0.23, 0.0, -0.113], [0.23, 0.0, 0.113], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.206, -0.07, -0.002], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [-0.1, -0.03, 0.0], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.019, 0.0, 0.074], [-0.019, 0.0, 0.128], [-0.074, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.052, 0.0, -0.097], [-0.037, 0.0, -0.097], [-0.029, 0.0, -0.127], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042]], dtype=np.float32)
        sw_anchor_2d = np.array([[650, 312], [630, 306], [907, 443], [814, 291], [599, 349], [501, 386], [965, 359], [649, 355], [635, 346], [930, 335], [843, 467], [702, 339], [718, 321], [930, 322], [727, 346], [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344], [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258], [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]], dtype=np.float32)
        sw_anchor_3d = np.array([[-0.035, -0.018, -0.01], [-0.057, -0.018, -0.01], [0.217, -0.0, -0.027], [-0.014, -0.0, 0.156], [-0.023, 0.0, -0.065], [-0.014, -0.0, -0.156], [0.234, -0.05, -0.002], [0.0, -0.0, -0.042], [-0.014, -0.0, -0.042], [0.206, -0.055, -0.002], [0.217, -0.0, -0.07], [0.025, -0.014, -0.011], [-0.014, -0.0, 0.042], [0.206, -0.07, -0.002], [0.049, -0.016, -0.011], [-0.029, -0.0, -0.127], [-0.019, -0.0, 0.128], [0.23, -0.0, 0.07], [0.217, -0.0, 0.07], [-0.052, -0.0, -0.097], [-0.175, -0.0, -0.015], [0.23, -0.0, -0.07], [-0.019, -0.0, 0.074], [0.23, -0.0, 0.113], [-0.0, -0.025, 0.24], [-0.0, -0.0, -0.015], [-0.074, -0.0, 0.128], [-0.074, -0.0, 0.074], [0.23, -0.0, -0.113], [0.243, -0.104, 0.0]], dtype=np.float32)
        sw2_anchor_2d = np.array([[15, 300],[1269, 180],[635, 143],[434, 274],[421, 240],[273, 320],[565, 266],[844, 206],[468, 543],[1185, 466],[565, 506],[569, 530],[741, 491],[1070, 459],[1089, 480],[974, 220],[941, 184],[659, 269],[650, 299],[636, 210],[620, 193]], dtype=np.float32)
        sw2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.017, -0.000, -0.092],[-0.052, -0.000, -0.097],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.042],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, -0.000, -0.027],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[0.025, -0.014, -0.011],[0.025, 0.000, -0.015],[-0.035, -0.018, -0.010],[-0.057, -0.018, -0.010]], dtype=np.float32)
        se2_anchor_2d = np.array([[48, 216],[1269, 320],[459, 169],[853, 528],[143, 458],[244, 470],[258, 451],[423, 470],[741, 500],[739, 516],[689, 176],[960, 301],[828, 290],[970, 264],[850, 254]], dtype=np.float32)
        se2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[0.230, -0.000, -0.113],[0.230, 0.000, -0.070],[0.217, -0.000, -0.070],[0.217, -0.000, -0.027],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[-0.100, -0.030, 0.000],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074]], dtype=np.float32)
        nw2_anchor_2d = np.array([[1268, 328],[1008, 419],[699, 399],[829, 373],[641, 325],[659, 310],[783, 30],[779, 113],[775, 153],[994, 240],[573, 226],[769, 265],[686, 284],[95, 269],[148, 375],[415, 353],[286, 349],[346, 320],[924, 360],[590, 324]], dtype=np.float32)
        nw2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],[-0.052, -0.000, -0.097],[-0.057, -0.018, -0.010],[-0.035, -0.018, -0.010],[0.243, -0.104, 0.000],[0.206, -0.070, -0.002],[0.206, -0.055, -0.002],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.170, -0.000, -0.015],[0.025, -0.014, -0.011],[-0.000, -0.025, 0.240],[-0.080, -0.000, 0.156],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.128],[-0.029, -0.000, -0.127],[-0.100, -0.030, 0.000]], dtype=np.float32)
        ne2_anchor_2d = np.array([[1035, 95],[740, 93],[599, 16],[486, 168],[301, 305],[719, 225],[425, 349],[950, 204],[794, 248],[844, 203],[833, 175],[601, 275],[515, 301],[584, 244],[503, 266]], dtype=np.float32)
        ne2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.080, -0.000, 0.156],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128]], dtype=np.float32)

        anchor_definitions = {
            'NE': {'path': 'NE.png', '2d': ne_anchor_2d, '3d': ne_anchor_3d},
            'NW': {'path': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png', '2d': nw_anchor_2d, '3d': nw_anchor_3d},
            'SE': {'path': 'SE.png', '2d': se_anchor_2d, '3d': se_anchor_3d},
            'SW': {'path': 'Anchor_B.png', '2d': sw_anchor_2d, '3d': sw_anchor_3d},
            'SW2': {'path': 'SW2.png', '2d': sw2_anchor_2d, '3d': sw2_anchor_3d},
            'SE2': {'path': 'SE2.png', '2d': se2_anchor_2d, '3d': se2_anchor_3d},
            'NW2': {'path': 'NW2.png', '2d': nw2_anchor_2d, '3d': nw2_anchor_3d},
            'NE2': {'path': 'NE2.png', '2d': ne2_anchor_2d, '3d': ne2_anchor_3d},
        }

        self.viewpoint_anchors = {}
        for viewpoint, data in anchor_definitions.items():
            if not os.path.exists(data['path']):
                raise FileNotFoundError(f"Required anchor image not found: {data['path']}")

            anchor_image_bgr = cv2.resize(cv2.imread(data['path']), (self.camera_width, self.camera_height))
            anchor_features = self._extract_features_sp(anchor_image_bgr)
            anchor_keypoints = anchor_features['keypoints'][0].cpu().numpy()

            sp_tree = cKDTree(anchor_keypoints)
            distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0)
            valid_mask = distances != np.inf

            self.viewpoint_anchors[viewpoint] = {
                'features': anchor_features,
                'map_3d': {idx: pt for idx, pt in zip(indices[valid_mask], data['3d'][valid_mask])}
            }
        print("   ...anchor data initialized.")

    def run(self):
        """Updated main loop that handles timestamped frames"""
        frame_id = 0
        while self.running:
            if not self.processing_queue.empty():
                # Receive (frame, t_capture) tuple
                frame_data = self.processing_queue.get()
                
                # Handle both old and new formats for compatibility
                if isinstance(frame_data, tuple) and len(frame_data) == 2:
                    frame, t_capture = frame_data
                else:
                    # Fallback for old format
                    frame = frame_data
                    t_capture = time.time()
                
                result = self._process_frame(frame, frame_id, t_capture)
                frame_id += 1

                # Log results with capture time
                self.all_poses_log.append({
                    'frame': result.frame_id, 
                    'success': result.pose_success,
                    'position': result.position.tolist() if result.position is not None else None,
                    'quaternion': result.quaternion.tolist() if result.quaternion is not None else None,
                    'kf_position': result.kf_position.tolist() if result.kf_position is not None else None,
                    'kf_quaternion': result.kf_quaternion.tolist() if result.kf_quaternion is not None else None,
                    'num_inliers': result.num_inliers, 
                    'viewpoint_used': result.viewpoint_used,
                    'capture_time': t_capture
                })
            else:
                time.sleep(0.001)

    def _process_frame(self, frame: np.ndarray, frame_id: int, t_capture: float) -> ProcessingResult:
        """Updated processing with timestamp-aware UKF updates"""
        result = ProcessingResult(frame_id=frame_id, frame=frame.copy(), pose_success=False, capture_time=t_capture)

        # 1. Object Detection using YOLO
        bbox = self._yolo_detect(frame)
        result.bbox = bbox

        # 2. Feature Matching and Pose Estimation
        best_pose = self._estimate_pose_with_temporal_consistency(frame, bbox)

        # 3. Pre-filtering: Check if the new pose measurement is valid   ADDed the 20250816
        is_valid = False
        if best_pose:
            orientation_valid = True
            if self.last_orientation is not None:
                angle_diff = math.degrees(self.quaternion_angle_diff(self.last_orientation, best_pose.quaternion))
                if angle_diff > self.ORI_MAX_DIFF_DEG:
                    orientation_valid = False
                    print(f"🚫 Frame {frame_id}: Rejected (Orientation Jump: {angle_diff:.1f}° > {self.ORI_MAX_DIFF_DEG}°)")
            
            # Coverage score validation (replaces simple matching ratio)
            coverage_valid = best_pose.coverage_score >= 0.25  # Minimum acceptable coverage
            if not coverage_valid:
                print(f"🚫 Frame {frame_id}: Rejected (Coverage Score: {best_pose.coverage_score:.2f} < 0.25)")
            
            # Combined validation: orientation is stable AND coverage is sufficient
            is_valid = orientation_valid and coverage_valid
            
            if is_valid:
                print(f"✅ Frame {frame_id}: Accepted (Orientation: {orientation_valid}, "
                      f"Coverage: {best_pose.coverage_score:.2f}, "
                      f"Inliers: {best_pose.inliers}/{best_pose.total_matches})")

        # 4. Kalman Filter Update with timestamp
        if is_valid and best_pose:
            self.rejected_consecutive_frames_count = 0
            result.position, result.quaternion = best_pose.position, best_pose.quaternion
            result.num_inliers, result.pose_success = best_pose.inliers, True
            result.viewpoint_used = best_pose.viewpoint
            self.last_orientation = best_pose.quaternion

            # R_adaptive = R_base * (1 + γ * e_reproj / max(n_inlier, 1))
            base_pos_noise = 1e-4
            base_quat_noise = 1e-4
            gamma = 1.0
            noise_scale = 1.0 + gamma * best_pose.reprojection_error / max(best_pose.inliers, 1)
            
            R = np.eye(7)
            R[0:3, 0:3] *= base_pos_noise * noise_scale
            R[3:7, 3:7] *= base_quat_noise * noise_scale

            t_now = time.monotonic()

            with self.pose_data_lock:
                kf_pos, kf_quat = self.kf.update_with_timestamp(
                    best_pose.position, 
                    best_pose.quaternion, 
                    t_meas=t_capture,
                    R=R,
                    t_now=t_now
                )
                result.kf_position, result.kf_quaternion = kf_pos, kf_quat
        else:
            # Handle rejection
            self.rejected_consecutive_frames_count += 1
            if self.rejected_consecutive_frames_count >= self.MAX_REJECTED_FRAMES:
                print(f"⚠️ Exceeded {self.MAX_REJECTED_FRAMES} consecutive rejections. Re-initializing KF.")
                with self.pose_data_lock: 
                    self.kf.initialized = False
                    self.kf.t_state = None  # Reset timestamp too
                self.last_orientation = None
                self.current_best_viewpoint = None
                self.needs_full_eval = True
                self.rejected_consecutive_frames_count = 0

        return result

    def _estimate_pose_with_temporal_consistency(self, frame: np.ndarray, bbox: Optional[Tuple]) -> Optional[PoseData]:
        """
        Adaptive Viewpoint Selection and Matching (Algorithm 1 in paper).
        
        Paper Algorithm 1:
          1. Evaluate ALL templates, compute score = λ₁·inliers + λ₂·coverage for each.
          2. Select best scoring viewpoint.
          3. If v_t == v_{t-1} AND best_score > θ_reuse → reuse (skip full eval next frame).
          4. Else → re-evaluate all templates next frame.
        
        Optimization: When reuse condition was met last frame (needs_full_eval=False),
        try previous viewpoint first. If score > θ_reuse, skip full evaluation.
        """
        LAMBDA_INLIERS = 1.0
        LAMBDA_COVERAGE = 15.0

        def compute_score(pose: PoseData) -> float:
            return LAMBDA_INLIERS * pose.inliers + LAMBDA_COVERAGE * pose.coverage_score

        # --- Reuse path: try previous viewpoint first if stable (Algorithm 1, line 11-12) ---
        if self.current_best_viewpoint and not self.needs_full_eval:
            pose_data = self._solve_for_viewpoint(frame, self.current_best_viewpoint, bbox)
            if pose_data:
                score = compute_score(pose_data)
                if score > self.theta_reuse:
                    # Reuse previous viewpoint — skip full evaluation
                    return pose_data
            # Quality dropped below θ_reuse — fall through to full evaluation
            self.needs_full_eval = True

        # --- Full evaluation: evaluate ALL templates (Algorithm 1, lines 2-10) ---
        all_poses = []
        for viewpoint in self.viewpoint_anchors.keys():
            pose_data = self._solve_for_viewpoint(frame, viewpoint, bbox)
            if pose_data:
                all_poses.append(pose_data)

        if not all_poses:
            return None

        # Select best viewpoint by score (Algorithm 1, lines 7-10)
        best_pose = max(all_poses, key=lambda p: compute_score(p))
        best_score = compute_score(best_pose)

        # Check reuse condition (Algorithm 1, lines 11-14)
        if best_pose.viewpoint == self.current_best_viewpoint and best_score > self.theta_reuse:
            # Same viewpoint, good quality → allow reuse next frame
            self.needs_full_eval = False
        else:
            # Viewpoint changed or quality insufficient → re-evaluate next frame
            self.needs_full_eval = True

        self.current_best_viewpoint = best_pose.viewpoint
        print(f"🎯 Selected viewpoint: {best_pose.viewpoint} "
              f"(inliers={best_pose.inliers}, "
              f"coverage={best_pose.coverage_score:.2f}, "
              f"score={best_score:.1f}, "
              f"reuse={'yes' if not self.needs_full_eval else 'no'})")
        return best_pose

    def _solve_for_viewpoint(self, frame: np.ndarray, viewpoint: str, bbox: Optional[Tuple]) -> Optional[PoseData]:
        """Attempts to calculate the pose for a single given viewpoint."""
        anchor = self.viewpoint_anchors.get(viewpoint)
        if not anchor: return None

        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        if crop.size == 0: return None

        # Extract features from the current frame's crop
        frame_features = self._extract_features_sp(crop)

        # Send visualization data if enabled
        if self.args.show and self.visualization_queue.qsize() < 2:
            kpts = frame_features['keypoints'][0].cpu().numpy()
            self.visualization_queue.put({'kpts': kpts, 'crop': crop.copy()})

        # Match features between anchor and current frame
        with torch.no_grad():
            matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        if len(matches) < 6: return None

        # Build 2D-3D point correspondences for PnP
        points_3d, points_2d = [], []
        crop_offset = np.array([bbox[0], bbox[1]]) if bbox else np.array([0, 0])
        for anchor_idx, frame_idx in matches:
            if anchor_idx in anchor['map_3d']:
                points_3d.append(anchor['map_3d'][anchor_idx])
                points_2d.append(frame_features['keypoints'][0].cpu().numpy()[frame_idx] + crop_offset)
        if len(points_3d) < 6: return None

        # Solve for pose using solvePnPRansac
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(points_3d, dtype=np.float32),
                np.array(points_2d, dtype=np.float32),
                self.K, self.dist_coeffs, reprojectionError=8, confidence=0.95,
                iterationsCount=3000, flags=cv2.SOLVEPNP_EPNP
            )
            # Refine pose using only inliers
            if success and inliers is not None and len(inliers) > 4:
                rvec, tvec = cv2.solvePnPRefineVVS(
                    np.array(points_3d, dtype=np.float32)[inliers.flatten()],
                    np.array(points_2d, dtype=np.float32)[inliers.flatten()],
                    self.K, self.dist_coeffs, rvec, tvec
                )
        except cv2.error as e:
            return None

        if not success or inliers is None or len(inliers) < 4: return None

        # Convert rotation vector to quaternion and calculate reprojection error
        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        quaternion_raw = self._rotation_matrix_to_quaternion(R)
        # Ensure quaternion is in [x,y,z,w] order and normalized
        quaternion = normalize_quaternion(quaternion_to_xyzw(quaternion_raw))
        
        projected_points, _ = cv2.projectPoints(np.array(points_3d)[inliers.flatten()], rvec, tvec, self.K, self.dist_coeffs)
        error = np.mean(np.linalg.norm(np.array(points_2d)[inliers.flatten()].reshape(-1, 1, 2) - projected_points, axis=2))

        # Coverage score (spatial distribution of inlier correspondences)
        inlier_3d = np.array(points_3d, dtype=np.float32)[inliers.flatten()]
        coverage_score = self._compute_coverage_score(inlier_3d)

        return PoseData(position, quaternion, len(inliers), error, viewpoint, len(points_3d), coverage_score)

    def _compute_coverage_score(self, inlier_3d_points: np.ndarray) -> float:
        """
        Entropy-based spatial distribution score over 4 body-frame regions.
        Returns 0–1, where 1 = uniform distribution across all regions.
        
        score = (-sum p_r * ln(p_r)) / ln(4)
        """
        if len(inlier_3d_points) == 0:
            return 0.0

        # Classify inlier 3D points into 4 regions based on body-frame axes
        regions = {"front-right": 0, "front-left": 0, "back-right": 0, "back-left": 0}
        for pt in inlier_3d_points:
            # pt = [x, y, z] in body frame
            # x < 0 => front (nose direction), x > 0 => back (tail)
            # z > 0 => right wing, z < 0 => left wing
            if pt[0] < 0 and pt[2] > 0:
                regions["front-right"] += 1
            elif pt[0] < 0 and pt[2] < 0:
                regions["front-left"] += 1
            elif pt[0] >= 0 and pt[2] > 0:
                regions["back-right"] += 1
            elif pt[0] >= 0 and pt[2] <= 0:
                regions["back-left"] += 1

        total_points = sum(regions.values())
        if total_points == 0:
            return 0.0

        # Compute normalized entropy
        entropy_sum = 0.0
        for count in regions.values():
            if count > 0:
                proportion = count / total_points
                entropy_sum += proportion * np.log(proportion)

        # Normalize by log(4) so score is in [0, 1]
        coverage_score = -entropy_sum / np.log(4)
        return float(np.clip(coverage_score, 0.0, 1.0))

    def quaternion_angle_diff(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculates the angular difference in radians between two quaternions."""
        dot = np.dot(normalize_quaternion(q1), normalize_quaternion(q2))
        return 2 * math.acos(abs(min(1.0, max(-1.0, dot))))

    def _yolo_detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detects aircraft in the frame using YOLO, returning the bounding box."""
        # Map class name to id; for your model it's usually {0: 'iha'}
        names = getattr(self.yolo_model, "names", {0: "iha"})
        inv = {v: k for k, v in names.items()}
        target_id = inv.get("iha", 0)

        for conf_thresh in (0.90, 0.75, 0.65):
            results = self.yolo_model(
                frame,
                imgsz=640,
                conf=conf_thresh,
                iou=0.5, 
                max_det=5,
                classes=[target_id],
                verbose=False
            )
            if not results or len(results[0].boxes) == 0:
                continue

            # Choose largest box (area)
            boxes = results[0].boxes
            best = max(
                boxes,
                key=lambda b: float((b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
            )
            x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()
            return int(x1), int(y1), int(x2), int(y2)

        return None

    def _extract_features_sp(self, image_bgr: np.ndarray) -> Dict:
        """Extracts SuperPoint features from a single BGR image."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        """Returns the camera intrinsic matrix K."""
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([
             0.02692405, -0.03433880, 0.01104186, 0.00124234, -0.12498783
         ], dtype=np.float32)
        return K, dist_coeffs

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Converts a 3x3 rotation matrix to a quaternion (x, y, z, w)."""
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return np.array([qx, qy, qz, qw])

    def cleanup(self):
        """Saves the pose log to a JSON file if requested."""
        print("Shutting down...")
        self.running = False
        if self.args.save_output:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"pose_log_{time.strftime('%Y%m%d-%H%M%S')}.json")
            with open(filename, 'w') as f:
                json.dump(self.all_poses_log, f, indent=4)
            print(f"Pose log saved to {filename}")


# --------------------------------------------------------------------------------------------------
#  7. MAIN EXECUTION BLOCK
# --------------------------------------------------------------------------------------------------
def main():
    """Parses arguments, sets up queues and threads, and starts the application."""
    parser = argparse.ArgumentParser(description="VAPE - Real-time Pose Estimator with Timestamp Support")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', action='store_true', help='Use webcam as input.')
    group.add_argument('--video_file', type=str, help='Path to a video file.')
    group.add_argument('--image_dir', type=str, help='Path to a directory of images.')
    parser.add_argument('--save_output', action='store_true', help='Save the final pose data to a JSON file.')
    parser.add_argument('--show', action='store_true', help='Show keypoint detections in a separate window.')
    args = parser.parse_args()

    try:
        # Queues for inter-thread communication
        processing_queue = queue.Queue(maxsize=2)
        visualization_queue = queue.Queue(maxsize=2)
        pose_data_lock = threading.Lock()

        # Initialize enhanced UKF with timestamp support
        kf = UnscentedKalmanFilter()
        
        kf.set_rate_limits(max_rotation_dps=30.0, max_position_mps=1.5)
        
        main_thread = MainThread(processing_queue, visualization_queue, pose_data_lock, kf, args)
        processing_thread = ProcessingThread(processing_queue, visualization_queue, pose_data_lock, kf, args)

        print("Starting VAPE in enhanced multi-threaded mode with timestamp support...")
        main_thread.start()
        processing_thread.start()

        # Wait for the main thread to finish
        main_thread.join()

        # Cleanly shut down the processing thread
        print("Stopping processing thread...")
        processing_thread.running = False
        processing_thread.join()
        print("Exiting.")

    except (IOError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Process interrupted by user (Ctrl+C).")
    finally:
        print("✅ Process finished.")


if __name__ == '__main__':
    main()
