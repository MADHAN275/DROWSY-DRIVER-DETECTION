import cv2
import time
import datetime
import csv
import argparse
import os
from src.camera.video_capture import VideoStream
from src.detection.face_detector import FaceDetector
from src.detection.eye_aspect_ratio import EyeAspectRatio
from src.detection.blink_analyzer import BlinkAnalyzer
from src.detection.head_pose_estimator import HeadPoseEstimator
from src.alerts.alarm_system import AlarmSystem
from src.dashboard.live_monitor import LiveMonitor
from src.utils.constants import (
    EAR_THRESHOLD, EAR_TIME_THRESHOLD, 
    PITCH_THRESHOLD, PITCH_TIME_THRESHOLD,
    BLINK_MIN_THRESHOLD, BLINK_MAX_THRESHOLD,
    FRAME_WIDTH, FRAME_HEIGHT
)
from src.utils.helpers import setup_logger

logger = setup_logger("Main")

class DrowsinessDetectorApp:
    def __init__(self, output_file: str = "drowsiness_log.csv"):
        self.video_stream = None
        self.face_detector = None
        self.blink_analyzer = None
        self.head_pose_estimator = None
        self.alarm_system = None
        self.live_monitor = None
        
        # State tracking
        self.ear_start_time = None
        self.pitch_start_time = None
        self.is_drowsy = False
        self.app_start_time = time.time()
        
        # Logging
        self.output_file = output_file
        self.log_file = None
        self.csv_writer = None

    def initialize(self):
        """Initialize all subsystems."""
        logger.info("Initializing subsystems...")
        self.video_stream = VideoStream()
        self.face_detector = FaceDetector(max_num_faces=2) # Detect up to 2 to warn about multiple faces
        self.blink_analyzer = BlinkAnalyzer()
        self.head_pose_estimator = HeadPoseEstimator()
        self.alarm_system = AlarmSystem()
        self.live_monitor = LiveMonitor()
        
        # Reset app start time after initialization to ensure fair warmup
        self.app_start_time = time.time()
        
        # Setup CSV logging
        file_exists = os.path.isfile(self.output_file)
        self.log_file = open(self.output_file, 'a', newline='')
        self.csv_writer = csv.writer(self.log_file)
        if not file_exists:
            self.csv_writer.writerow(["Timestamp", "Event", "EAR", "Pitch", "BlinkRate"])
            
        logger.info("Initialization complete.")

    def run(self):
        """Main application loop."""
        try:
            while True:
                # 1. Capture Frame
                success, frame = self.video_stream.read()
                if not success:
                    continue

                # 2. Face Detection
                results = self.face_detector.process(frame)
                
                ear = 0.0
                pitch, yaw, roll = 0.0, 0.0, 0.0
                blinks_pm = 0
                
                face_detected = False
                multiple_faces = False
                
                # Check for face landmarks (Note: Tasks API uses face_landmarks)
                if results.face_landmarks:
                    if len(results.face_landmarks) > 1:
                        multiple_faces = True
                        cv2.putText(frame, "WARNING: Multiple faces detected", (10, 450), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    # Process the first face only for drowsiness
                    landmarks = self.face_detector.get_landmarks(results, FRAME_WIDTH, FRAME_HEIGHT)
                    
                    if landmarks is not None:
                        face_detected = True
                        
                        # 3. Calculate Metrics
                        # EAR
                        ear = EyeAspectRatio.get_avg_ear(landmarks)
                        
                        # Blink Rate
                        self.blink_analyzer.update(ear)
                        blinks_pm = self.blink_analyzer.get_blinks_per_minute()
                        
                        # Head Pose
                        pitch, yaw, roll = self.head_pose_estimator.get_pose(landmarks)
                        
                        # 4. Check Logic
                        self._check_drowsiness(ear, pitch, blinks_pm)
                        
                        # 5. Visualizer
                        # Draw landmarks (optional, maybe just eyes/face contour)
                        # For clean UI, maybe skip drawing mesh or make it subtle
                        pass
                else:
                    self._reset_state()
                    cv2.putText(frame, "NO DRIVER DETECTED", (150, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # 6. Dashboard & Alerts
                fps = 0 # Calculate real FPS if needed, or rely on update loop timing
                # Simple FPS calc
                fps = 1.0 / (time.time() - self.video_stream.prev_time + 0.0001) # Approx
                
                self.live_monitor.update(frame, ear, blinks_pm, pitch, yaw, roll, self.is_drowsy, fps)
                
                if self.is_drowsy:
                    self.alarm_system.trigger_alarm()
                    frame = self.alarm_system.draw_alert(frame)
                else:
                    self.alarm_system.stop_alarm()
                
                # Show Frame
                cv2.imshow("Drowsiness Detection System", frame)
                
                # Check Quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested...")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
        finally:
            self.cleanup()

    def _check_drowsiness(self, ear: float, pitch: float, blinks_pm: int):
        """
        Evaluate conditions for drowsiness.
        """
        current_time = time.time()
        drowsy_conditions = []

        # Condition 1: EAR low (Eyes closed)
        if ear < EAR_THRESHOLD:
            if self.ear_start_time is None:
                self.ear_start_time = current_time
            elif current_time - self.ear_start_time > EAR_TIME_THRESHOLD:
                drowsy_conditions.append("Eyes Closed")
        else:
            self.ear_start_time = None

        # Condition 2: Head nodding (Pitch down)
        # Note: Pitch > Threshold usually means looking down if Y is down.
        if pitch > PITCH_THRESHOLD:
            if self.pitch_start_time is None:
                self.pitch_start_time = current_time
            elif current_time - self.pitch_start_time > PITCH_TIME_THRESHOLD:
                drowsy_conditions.append("Head Nod")
        else:
            self.pitch_start_time = None

        # Condition 3: Abnormal Blink Rate
        # This is instantaneous, no time duration needed (already rolling window)
        # Note: We give a 60s warmup period before flagging low blink rate
        if time.time() - self.app_start_time > 60:
            if blinks_pm < BLINK_MIN_THRESHOLD:
                drowsy_conditions.append("Low Blink Rate")
        
        if blinks_pm > BLINK_MAX_THRESHOLD:
            # High blink rate (strain) can be flagged immediately
            drowsy_conditions.append("High Blink Rate")

        # Decision
        if drowsy_conditions:
            if not self.is_drowsy:
                self.is_drowsy = True
                self._log_event(drowsy_conditions, ear, pitch, blinks_pm)
        else:
            self.is_drowsy = False

    def _reset_state(self):
        """Reset timers if face lost."""
        self.ear_start_time = None
        self.pitch_start_time = None
        self.is_drowsy = False
        self.alarm_system.stop_alarm()

    def _log_event(self, reasons: list, ear: float, pitch: float, blinks: int):
        """Log drowsiness event to CSV."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reason_str = "; ".join(reasons)
        if self.csv_writer:
            self.csv_writer.writerow([timestamp, reason_str, f"{ear:.3f}", f"{pitch:.1f}", blinks])
            self.log_file.flush()
            logger.warning(f"Drowsiness detected: {reason_str}")

    def cleanup(self):
        """Release resources."""
        logger.info("Cleaning up...")
        if self.video_stream:
            self.video_stream.release()
        if self.face_detector:
            self.face_detector.close()
        if self.alarm_system:
            self.alarm_system.stop_alarm()
        if self.log_file:
            self.log_file.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drowsy Driver Detection System")
    parser.add_argument("--output", type=str, default="drowsiness_log.csv", help="Path to log file")
    args = parser.parse_args()
    
    app = DrowsinessDetectorApp(output_file=args.output)
    try:
        app.initialize()
        app.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
