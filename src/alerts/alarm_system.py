import pygame
import cv2
import os
import time
from typing import Tuple
from src.utils.constants import COLOR_RED, COLOR_WHITE, FRAME_WIDTH, FRAME_HEIGHT
from src.utils.helpers import setup_logger

logger = setup_logger("AlarmSystem")

class AlarmSystem:
    """
    Handles audio and visual alerts.
    """
    def __init__(self, sound_file: str = "assets/alarm.wav"):
        """
        Initialize the alarm system.
        
        Args:
            sound_file: Path to the wav file.
        """
        self.sound_file = sound_file
        
        # Initialize Pygame Mixer
        try:
            pygame.mixer.init()
            if os.path.exists(sound_file):
                self.sound = pygame.mixer.Sound(sound_file)
            else:
                logger.error(f"Sound file not found: {sound_file}")
                self.sound = None
        except Exception as e:
            logger.error(f"Failed to init audio: {e}")
            self.sound = None
            
        self.is_playing = False
        self.last_triggered = 0

    def trigger_alarm(self):
        """
        Play the alarm sound if not already playing.
        """
        if self.sound and not self.is_playing:
            self.sound.play(-1) # Loop indefinitely
            self.is_playing = True
            logger.warning("ALARM TRIGGERED!")

    def stop_alarm(self):
        """
        Stop the alarm sound.
        """
        if self.sound and self.is_playing:
            self.sound.stop()
            self.is_playing = False
            logger.info("Alarm stopped")

    def draw_alert(self, frame: cv2.Mat, message: str = "DROWSY ALERT!") -> cv2.Mat:
        """
        Overlay a visual alert on the frame.
        
        Args:
            frame: Current video frame.
            message: Alert text.
            
        Returns:
            Frame with alert overlay.
        """
        # Create a red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), COLOR_RED, -1)
        
        # Blend with original frame (alpha blending)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(message, font, 2, 3)[0]
        text_x = (FRAME_WIDTH - text_size[0]) // 2
        text_y = (FRAME_HEIGHT + text_size[1]) // 2
        
        cv2.putText(frame, message, (text_x, text_y), font, 2, COLOR_WHITE, 3)
        
        return frame
