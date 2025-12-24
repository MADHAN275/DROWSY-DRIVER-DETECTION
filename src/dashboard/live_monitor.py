import cv2
import numpy as np
from collections import deque
from src.utils.constants import (
    FRAME_WIDTH, COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_WHITE, COLOR_BLACK,
    EAR_THRESHOLD, BLINK_MIN_THRESHOLD, BLINK_MAX_THRESHOLD
)

class LiveMonitor:
    """
    Renders the real-time dashboard on the video frame.
    """
    def __init__(self):
        """
        Initialize dashboard elements.
        """
        # EAR Graph history
        self.ear_history = deque(maxlen=100)
        self.graph_height = 100
        self.graph_width = 200
        self.graph_x = FRAME_WIDTH - self.graph_width - 10
        self.graph_y = 10

    def update(self, frame: cv2.Mat, ear: float, blinks_pm: int, 
               pitch: float, yaw: float, roll: float, 
               is_drowsy: bool, fps: float):
        """
        Update the dashboard with new metrics.
        
        Args:
            frame: Video frame to draw on.
            ear: Current Eye Aspect Ratio.
            blinks_pm: Blinks per minute.
            pitch: Head pitch.
            yaw: Head yaw.
            roll: Head roll.
            is_drowsy: Boolean indicating drowsiness state.
            fps: Frames per second.
        """
        self.ear_history.append(ear)
        
        # 1. Draw Status
        status_color = COLOR_RED if is_drowsy else COLOR_GREEN
        status_text = "DROWSY" if is_drowsy else "ACTIVE"
        cv2.putText(frame, f"STATUS: {status_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 2. Draw Metrics Text
        y_offset = 60
        line_height = 25
        
        # EAR
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        y_offset += line_height
        
        # Blinks
        blink_color = COLOR_WHITE
        if blinks_pm < BLINK_MIN_THRESHOLD or blinks_pm > BLINK_MAX_THRESHOLD:
            blink_color = COLOR_YELLOW
        cv2.putText(frame, f"Blinks/min: {blinks_pm}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 1)
        y_offset += line_height
        
        # Head Pose
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        y_offset += line_height
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        y_offset += line_height
        cv2.putText(frame, f"Roll: {roll:.1f}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        y_offset += line_height
        
        # FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1)

        # 3. Draw EAR Graph
        self._draw_graph(frame)

    def _draw_graph(self, frame: cv2.Mat):
        """Draw the rolling EAR graph."""
        # Draw background
        cv2.rectangle(frame, 
                     (self.graph_x, self.graph_y), 
                     (self.graph_x + self.graph_width, self.graph_y + self.graph_height), 
                     (50, 50, 50), -1)
        
        if len(self.ear_history) < 2:
            return

        # Normalize points
        # Assuming EAR range mostly 0.15 to 0.4
        min_val = 0.15
        max_val = 0.4
        
        points = []
        for i, val in enumerate(self.ear_history):
            # Clamp value
            val = max(min_val, min(max_val, val))
            
            # X coordinate
            x = self.graph_x + int((i / 100) * self.graph_width)
            
            # Y coordinate (inverted because Y is down)
            normalized_h = (val - min_val) / (max_val - min_val)
            y = self.graph_y + self.graph_height - int(normalized_h * self.graph_height)
            points.append((x, y))

        # Draw lines
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i+1], COLOR_GREEN, 1)
            
        # Draw threshold line
        thresh_norm = (EAR_THRESHOLD - min_val) / (max_val - min_val)
        thresh_y = self.graph_y + self.graph_height - int(thresh_norm * self.graph_height)
        cv2.line(frame, 
                 (self.graph_x, thresh_y), 
                 (self.graph_x + self.graph_width, thresh_y), 
                 COLOR_RED, 1)
        
        cv2.putText(frame, "EAR History", (self.graph_x, self.graph_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
