try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import FaceLandmarker
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarkerOptions
    from mediapipe.tasks.python.vision import RunningMode
    print("Found FaceLandmarker in tasks!")
except ImportError as e:
    print(f"Could not import FaceLandmarker: {e}")
    import mediapipe.tasks
    print(dir(mediapipe.tasks))
