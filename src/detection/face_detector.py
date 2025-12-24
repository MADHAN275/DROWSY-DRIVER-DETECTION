import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Any
import os
from src.utils.helpers import setup_logger

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = setup_logger("FaceDetector")

class FaceDetector:
    """
    Wrapper for MediaPipe FaceMesh (via Tasks API) to detect facial landmarks.
    """
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Face Landmarker.
        """
        model_path = os.path.join("assets", "face_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please download it.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # Determine running mode
        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        # Note: Use VIDEO for sequential frames to enable tracking. 
        # If the pipeline assumes purely independent frames, IMAGE is safer but slower.
        # Given "live_monitor", likely we want VIDEO or LIVE_STREAM, but VIDEO is a good middle ground 
        # if we pass timestamps, or IMAGE if we don't.
        # The legacy static_image_mode=False implied video stream.
        # However, FaceLandmarker in VIDEO mode REQUIRES timestamps.
        # To avoid complicating the `process` call with timestamps if not passed, 
        # let's stick to IMAGE mode if it's fast enough, or check if we can handle VIDEO.
        # The legacy code didn't pass timestamps explicitly to `process`.
        # So let's strictly use IMAGE mode for simplicity unless performance is critical, 
        # or handle timestamps.
        # Actually, for real-time app, LIVE_STREAM is best but requires an async callback.
        # VIDEO mode is for offline video files usually.
        # IMAGE mode is simplest for a drop-in replacement where we just call process(frame).
        # Let's use IMAGE mode to be safe with the existing synchronous architecture.
        running_mode = vision.RunningMode.IMAGE

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence, # mapping similar concept
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        
        try:
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info("FaceDetector initialized (MediaPipe FaceLandmarker)")
        except Exception as e:
            logger.error(f"Failed to initialize FaceLandmarker: {e}")
            raise e

    def process(self, frame: np.ndarray) -> Any:
        """
        Process a frame and return face landmarks.
        
        Args:
            frame: BGR image from OpenCV.
            
        Returns:
            The processed results object from MediaPipe FaceLandmarker.
        """
        # MediaPipe Tasks requires mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect
        # Using IMAGE mode, so just detect(image)
        results = self.landmarker.detect(mp_image)
        
        return results

    def get_landmarks(self, results, frame_width: int, frame_height: int) -> Optional[np.ndarray]:
        """
        Extracts landmarks as a numpy array of (x, y) coordinates for the first detected face.
        
        Args:
            results: MediaPipe FaceLandmarkerResult object.
            frame_width: Width of the image.
            frame_height: Height of the image.
            
        Returns:
            np.ndarray: Array of shape (468, 2) containing (x, y) coordinates, or None if no face.
        """
        if not results.face_landmarks:
            return None
        
        # We assume single face monitoring for the driver
        # face_landmarks is a list of lists of NormalizedLandmark
        face_landmarks_list = results.face_landmarks[0]
        
        # Convert normalized coordinates to pixel coordinates
        # Note: FaceLandmarker returns 478 landmarks (468 mesh + 10 iris).
        # We slice to 468 to match expected legacy behavior unless 478 is handled downstream.
        # The constants.py usually defines 468 points.
        landmarks = np.array([
            (int(lm.x * frame_width), int(lm.y * frame_height)) 
            for lm in face_landmarks_list[:468]
        ])
        
        return landmarks
        
    def close(self):
        """Release resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
