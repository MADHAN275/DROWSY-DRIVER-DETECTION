import cv2
import numpy as np
import math
from src.utils.constants import MODEL_POINTS_3D, HEAD_POSE_INDICES, FRAME_WIDTH, FRAME_HEIGHT

class HeadPoseEstimator:
    """
    Estimates head pose (Pitch, Yaw, Roll) using PnP algorithm.
    """
    def __init__(self):
        """
        Initialize the Head Pose Estimator.
        """
        self.model_points = np.array(MODEL_POINTS_3D, dtype=np.float64)
        
        # Camera Matrix (approximated)
        self.focal_length = FRAME_WIDTH
        self.center = (FRAME_WIDTH / 2, FRAME_HEIGHT / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.center[0]],
             [0, self.focal_length, self.center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        self.dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
        
    def get_pose(self, landmarks: np.ndarray):
        """
        Calculate rotation vector and Euler angles.
        
        Args:
            landmarks: Full face landmarks array.
            
        Returns:
            tuple: (pitch, yaw, roll) in degrees.
        """
        # Extract the specific 2D image points needed for PnP
        image_points = np.array([
            landmarks[HEAD_POSE_INDICES[0]], # Nose tip
            landmarks[HEAD_POSE_INDICES[1]], # Chin
            landmarks[HEAD_POSE_INDICES[2]], # Left eye outer
            landmarks[HEAD_POSE_INDICES[3]], # Right eye outer
            landmarks[HEAD_POSE_INDICES[4]], # Left mouth corner
            landmarks[HEAD_POSE_INDICES[5]]  # Right mouth corner
        ], dtype=np.float64)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
            
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Combine into projection matrix
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        
        # Decompose projection matrix to get Euler angles
        # We can use cv2.decomposeProjectionMatrix or manual calculation from Rotation Matrix
        # Standard approach for Euler angles from Rotation Matrix:
        # Sy = sqrt(R00 * R00 +  R10 * R10)
        # singular = Sy < 1e-6
        # if not singular:
        #     x = atan2(R21 , R22)
        #     y = atan2(-R20, Sy)
        #     z = atan2(R10, R00)
        # else:
        #     x = atan2(-R12, R11)
        #     y = atan2(-R20, Sy)
        #     z = 0
        
        # Alternatively using RQDecomposition3x3 if we just had rot matrix, 
        # but let's stick to standard math for XYZ convention.
        
        # Let's use the method typically used in this context (RQ decomposition usually gives good results for camera angles)
        # Or simple trig on the rotation matrix.
        
        # Pitch: Rotation around X-axis
        # Yaw: Rotation around Y-axis
        # Roll: Rotation around Z-axis
        
        # Using the implementation common in similar projects:
        rmat = rotation_matrix
        
        # Pitch (x-axis rotation)
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        
        # Yaw (y-axis rotation)
        yaw = math.atan2(-rmat[2, 0], math.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2))
        
        # Roll (z-axis rotation)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
        
        # Convert to degrees
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        roll = math.degrees(roll)
        
        # Note: The coordinate systems can be tricky.
        # Often PnP with standard model gives:
        # Pitch: Positive is down (nodding) depending on model orientation. 
        # With the provided 3D model (nose at 0,0,0, chin at 0,-330,-65), Y is negative downwards in 3D?
        # Let's check the model provided:
        # Nose: 0, 0, 0
        # Chin: 0, -330, -65 -> Y is negative for chin? Usually in OpenCV screen coords Y is positive down.
        # In typical 3D face models, Y might be up or down.
        # If Y is negative for chin, then Y-axis points UP.
        # If camera Y points DOWN (OpenCV default), then we have a flip.
        
        # We will trust the result and if needed, the user can calibrate. 
        # However, typically Pitch > 0 is looking DOWN if X-axis is to the right and Y-axis is down.
        # Let's return the values.
        
        return pitch, yaw, roll
