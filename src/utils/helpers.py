import math
import logging
from typing import Tuple

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculates the Euclidean distance between two 2D points.
    
    Args:
        point1: (x, y) coordinates of the first point.
        point2: (x, y) coordinates of the second point.
        
    Returns:
        float: The Euclidean distance.
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def setup_logger(name: str = "DrowsinessDetector") -> logging.Logger:
    """
    Sets up a logger with a specific format.
    
    Args:
        name: Name of the logger.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
