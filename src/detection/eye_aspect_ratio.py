import numpy as np
from src.utils.constants import LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from src.utils.helpers import calculate_distance

class EyeAspectRatio:
    """
    Calculates the Eye Aspect Ratio (EAR) for drowsiness detection.
    """
    
    @staticmethod
    def calculate_ear(landmarks: np.ndarray, indices: list) -> float:
        """
        Calculates EAR for a single eye.
        
        Args:
            landmarks: Array of (x, y) facial landmarks.
            indices: List of 6 landmark indices for the eye.
                     [p1, p2, p3, p4, p5, p6] corresponds to:
                     p1, p4: Outer corners (horizontal)
                     p2, p6: Top/Bottom pair 1 (vertical)
                     p3, p5: Top/Bottom pair 2 (vertical)
                     
                     Note on indices mapping based on prompt:
                     Left: [33, 160, 158, 133, 153, 144]
                     33 (p1, left corner), 133 (p4, right corner)
                     160 (p2), 144 (p6)
                     158 (p3), 153 (p5)
                     
                     Formula: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        
        Returns:
            float: The calculated EAR value.
        """
        # Extract points
        p1 = landmarks[indices[0]]
        p2 = landmarks[indices[1]]
        p3 = landmarks[indices[2]]
        p4 = landmarks[indices[3]]
        p5 = landmarks[indices[4]]
        p6 = landmarks[indices[5]]

        # Calculate vertical distances
        # |p2 - p6|
        vertical_1 = calculate_distance(p2, p6)
        # |p3 - p5|
        vertical_2 = calculate_distance(p3, p5)

        # Calculate horizontal distance
        # |p1 - p4|
        horizontal = calculate_distance(p1, p4)

        if horizontal == 0:
            return 0.0

        # EAR formula
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    @staticmethod
    def get_avg_ear(landmarks: np.ndarray) -> float:
        """
        Calculates the average EAR for both eyes.
        
        Args:
            landmarks: Array of facial landmarks.
            
        Returns:
            float: Average EAR.
        """
        left_ear = EyeAspectRatio.calculate_ear(landmarks, LEFT_EYE_INDICES)
        right_ear = EyeAspectRatio.calculate_ear(landmarks, RIGHT_EYE_INDICES)
        return (left_ear + right_ear) / 2.0
