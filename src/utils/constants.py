from typing import List

# --- Camera Configuration ---
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
FPS_LIMIT: int = 30

# --- Detection Thresholds ---
EAR_THRESHOLD: float = 0.25         # Eye Aspect Ratio threshold for drowsiness
EAR_TIME_THRESHOLD: float = 2.5     # Seconds eyes must be closed to trigger alarm
BLINK_MIN_THRESHOLD: int = 10       # Minimum blinks per minute (fatigue)
BLINK_MAX_THRESHOLD: int = 30       # Maximum blinks per minute (strain/nervousness)
PITCH_THRESHOLD: float = 20.0       # Degrees downward for head nod
PITCH_TIME_THRESHOLD: float = 2.0   # Seconds head must be down to trigger alarm

# --- MediaPipe Landmark Indices ---
# Left eye indices: [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES: List[int] = [33, 160, 158, 133, 153, 144]
# Right eye indices: [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES: List[int] = [362, 385, 387, 263, 373, 380]

# --- 3D Model Points for Head Pose (Standard Face Model) ---
# Used for PnP solver
# Format: (x, y, z)
# Nose tip, Chin, Left Eye Left Corner, Right Eye Right Corner, Left Mouth Corner, Right Mouth Corner
MODEL_POINTS_3D = [
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left Mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
]

# Corresponding MediaPipe Indices for the above 3D points
# Nose tip (1), Chin (152), Left Eye Left Corner (263 - wait, 263 is right eye outer, 33 is left eye outer)
# Let's verify standard MediaPipe canonical face mesh indices.
# 1: Nose tip
# 199: Chin (or 152) -> 152 is commonly used for chin bottom
# 33: Left eye outer corner
# 263: Right eye outer corner
# 61: Left mouth corner
# 291: Right mouth corner
HEAD_POSE_INDICES: List[int] = [1, 152, 33, 263, 61, 291]

# --- Colors (BGR) ---
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
