import cv2
import time
from typing import Tuple, Generator, Optional
from src.utils.constants import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS_LIMIT
from src.utils.helpers import setup_logger

logger = setup_logger("VideoCapture")

class VideoStream:
    """
    Handles video capture from the webcam with FPS limiting and resizing.
    """
    def __init__(self, src: int = CAMERA_INDEX):
        """
        Initialize the video stream.
        
        Args:
            src: Camera index (default 0).
        """
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            logger.error(f"Failed to open camera with index {src}")
            raise RuntimeError(f"Could not start video stream on source {src}")
            
        # Set resolution
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Camera started: {self.width}x{self.height} @ {FPS_LIMIT} FPS limit")
        
        self.prev_time = 0
        self.fps_limit = FPS_LIMIT

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple[bool, np.ndarray]: (Success, Frame)
        """
        # Simple FPS limiting logic
        time_elapsed = time.time() - self.prev_time
        if time_elapsed < 1./self.fps_limit:
            return False, None
            
        self.prev_time = time.time()
        
        ret, frame = self.stream.read()
        if not ret:
            logger.warning("Failed to grab frame")
            return False, None
            
        # Ensure correct size (some cameras might ignore set prop)
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
        return True, frame

    def release(self):
        """
        Release the video resource.
        """
        self.stream.release()
        logger.info("Camera released")
