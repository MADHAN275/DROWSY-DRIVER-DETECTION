import time
from collections import deque
from src.utils.constants import EAR_THRESHOLD

class BlinkAnalyzer:
    """
    Analyzes blink rate over time.
    """
    def __init__(self, history_seconds: int = 60):
        """
        Initialize the blink analyzer.
        
        Args:
            history_seconds: Window size in seconds for blink counting.
        """
        self.blink_timestamps = deque()
        self.history_seconds = history_seconds
        
        # State for blink detection
        self.eye_closed = False
        
    def update(self, ear: float):
        """
        Update blink logic with current EAR.
        
        Args:
            ear: Current average EAR.
        """
        current_time = time.time()
        
        # Simple finite state machine for blink detection
        # Eye closes (EAR drops below threshold)
        if ear < EAR_THRESHOLD:
            self.eye_closed = True
            
        # Eye opens (EAR rises above threshold) -> Blink completed
        elif ear >= EAR_THRESHOLD and self.eye_closed:
            self.eye_closed = False
            self.blink_timestamps.append(current_time)
            
        # Clean up old timestamps
        self._cleanup(current_time)
        
    def _cleanup(self, current_time: float):
        """Remove timestamps older than the history window."""
        while self.blink_timestamps and (current_time - self.blink_timestamps[0] > self.history_seconds):
            self.blink_timestamps.popleft()
            
    def get_blinks_per_minute(self) -> int:
        """
        Calculate blinks per minute based on the rolling window.
        
        Returns:
            int: Blinks count in the last minute.
        """
        # Since our window is exactly 1 minute (by default), 
        # the length of the deque is the BPM.
        # If history_seconds is not 60, we would need to scale.
        count = len(self.blink_timestamps)
        if self.history_seconds == 60:
            return count
        else:
            return int(count * (60 / self.history_seconds))
